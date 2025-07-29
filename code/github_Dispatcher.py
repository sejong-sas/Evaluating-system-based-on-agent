# github_Dispatcher.py
# High-Recall 2-Pass (evidence 태그만 수집 → 요약) / 4개 그룹 병렬 구조와 동일 메커니즘
# - OpenAI Chat Completions 사용
# - response_format={"type":"json_object"} 강제
# - 시스템/유저 메시지에 'json' 명시
# - __sources 제거, __evidence 만 유지 (예: ["[readme]", "[files]", "[py_files/path/to/file.py]"])

import os
import json
from typing import Dict, List, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
_client = OpenAI(api_key=_api_key)

# ===== 16개 평가항목 라벨 =====
LABELS = {
    "1-1": "1-1 (가중치 Weights)",
    "1-2": "1-2 (코드 Code)",
    "1-3": "1-3 (라이선스 License)",
    "1-4": "1-4 (논문 Paper)",
    "1-5": "1-5 (아키텍처 Architecture)",
    "1-6": "1-6 (토크나이저 Tokenizer)",
    "2-1": "2-1 (하드웨어 Hardware)",
    "2-2": "2-2 (소프트웨어 Software)",
    "2-3": "2-3 (API)",
    "3-1": "3-1 (사전학습 Pre-training)",
    "3-2": "3-2 (파인튜닝 Fine-tuning)",
    "3-3": "3-3 (강화학습 Reinforcement Learning)",
    "4-1": "4-1 (사전학습 데이터 Pre-training Data)",
    "4-2": "4-2 (파인튜닝 데이터 Fine-tuning Data)",
    "4-3": "4-3 (강화학습 데이터 Reinforcement Learning Data)",
    "4-4": "4-4 (데이터 필터링 Data Filtering)",
}

# ===== 각 항목의 구체 설명 =====
EVAL_DESCRIPTIONS = {
    LABELS["1-1"]: "모델 가중치의 공개 여부, 위치, 접근 방식, 누구나 다운로드 가능한지에 관련된 모든 내용",
    LABELS["1-2"]: "모델 훈련 및 실행을 위한 코드가 공개되었는지, 어떤 부분이 공개되었는지에 관련된 모든 내용",
    LABELS["1-3"]: "라이선스의 존재 여부, 종류, 허용된 권리(사용, 수정, 배포, 상업적 이용)에 관련된 모든 내용",
    LABELS["1-4"]: "모델과 관련된 공식 논문, 기술 보고서, 블로그 등 문서의 존재와 링크에 관련된 모든 내용",
    LABELS["1-5"]: "모델 아키텍처(레이어 수, 하이퍼파라미터 등)와 구조 설계의 세부 정보에 관련된 모든 내용",
    LABELS["1-6"]: "어떤 토크나이저를 사용하는지, 이름과 구조, 다운로드 가능 여부에 관련된 모든 내용",
    LABELS["2-1"]: "모델 훈련에 사용된 하드웨어 종류(H100, TPU 등), 수량, 계산 자원 규모에 관련된 모든 내용",
    LABELS["2-2"]: "훈련에 사용된 소프트웨어(프레임워크, 라이브러리 등)의 종류, 버전, 설정에 관련된 모든 내용",
    LABELS["2-3"]: "모델이 접근 가능한 API의 존재 여부, 문서 링크, 사용 예제, 공개 여부에 관련된 모든 내용",
    LABELS["3-1"]: "사전학습(pre-training) 시 사용된 방법론, 절차, 데이터 흐름, 하이퍼파라미터 설정 등에 관련된 모든 내용",
    LABELS["3-2"]: "파인튜닝 방식, 목적, 데이터 사용 여부, 재현 가능한 파이프라인 존재 여부에 관련된 모든 내용",
    LABELS["3-3"]: "RLHF, DPO 등 강화학습 알고리즘 사용 여부, 구체적인 방식과 절차, 설정값 등에 관련된 모든 내용",
    LABELS["4-1"]: "사전학습에 사용된 데이터의 종류, 수량, 출처, 사용 범위 및 구성 방식에 관련된 모든 내용",
    LABELS["4-2"]: "파인튜닝에 사용된 데이터셋의 출처, 구성, 데이터 예시, 공개 여부 등에 관련된 모든 내용",
    LABELS["4-3"]: "강화학습에 사용된 데이터셋의 구성, 접근 가능 여부, 출처, 생성 방식에 관련된 모든 내용",
    LABELS["4-4"]: "데이터 필터링 또는 정제 방법, 사용된 기준, 필터링 과정과 그 영향에 관련된 모든 내용",
}

# ===== 4개 그룹 분할 (HF 디스패처와 동일) =====
ITEM_GROUPS: List[List[str]] = [
    ["1-1", "1-2", "1-3", "1-4"],
    ["1-5", "1-6", "2-1", "2-2"],
    ["2-3", "3-1", "3-2", "3-3"],
    ["4-1", "4-2", "4-3", "4-4"],
]

# ===== High-Recall & Model 설정 =====
CHUNK_CHARS = 60_000            # GitHub README/코드가 길 수 있어 크게
CHUNK_OVERLAP = 2_000
EVIDENCE_LIMIT_PER_KEY = 300
MODEL_NAME = os.getenv("OPENAI_MODEL_GH_DISPATCHER", "gpt-4o")
TEMPERATURE = 0

# ===== 유틸 =====
def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _chunk_text(s: str, chunk_chars: int, overlap: int) -> List[str]:
    chunks, n, i = [], len(s), 0
    while i < n:
        end = min(i + chunk_chars, n)
        chunks.append(s[i:end])
        if end == n: break
        i = end - overlap if end - overlap > i else end
    return chunks

def _dedup_list(seq: List[str], limit: int) -> List[str]:
    seen, out = set(), []
    for x in seq:
        x_norm = (x or "").strip()
        if x_norm and x_norm not in seen:
            seen.add(x_norm)
            out.append(x_norm)
            if len(out) >= limit: break
    return out

def _group_desc_map(group_ids: List[str]) -> Dict[str, str]:
    return {LABELS[k]: EVAL_DESCRIPTIONS[LABELS[k]] for k in group_ids}

# ===== 프롬프트 (json 단어를 반드시 포함) =====
_BASE_RECALL_SYS = """
당신은 GitHub 저장소에서 AI 모델 개방성 평가 정보를 추출하는 전문가입니다.
오직 제공된 payload(원문)만 사용하세요. 외부 자료 금지.
반드시 json 객체(JSON object)만 반환하세요. 텍스트/설명/백틱을 덧붙이지 마세요.
"""

def _build_recall_inst(group_ids: List[str]) -> str:
    """
    evidence 태그만 배열로 수집하도록 지시:
      허용 태그 예시:
        [repo], [readme], [license_files], [files],
        [py_files/path/to/file.py]
    """
    desc_map = _group_desc_map(group_ids)
    example = {
        LABELS[k]: [
            "[readme]", "[files]", "[license_files]", "[py_files/path/to/train.py]"
        ] for k in group_ids
    }
    return (
        "이번 그룹 항목 정의:\n"
        + _json_dumps(desc_map)
        + "\n각 키에 대해 evidence '태그'만 배열로 반환하세요.\n"
        + "허용되는 태그는 대괄호[]로 둘러싸인 섹션 이름입니다.\n"
        + "예: [readme], [files], [license_files], [py_files/xxx.py], [repo]\n"
        + "예시 스키마:\n"
        + _json_dumps(example)
    )

_BASE_SUMMARY_SYS = """
당신은 GitHub evidence 태그만 사용해 각 항목별 '길고 상세한' 요약을 작성하는 전문가입니다.
오직 제공된 evidence 태그만 사용하세요. 외부 자료 금지.
반드시 json 객체(JSON object)만 반환하세요. 텍스트/설명/백틱을 덧붙이지 마세요.
- 각 항목 값은 하나의 긴 문자열입니다.
- 같은 의미의 중복은 줄이되, 고유 정보는 최대한 포함하세요.
"""

def _build_summary_inst(group_ids: List[str]) -> str:
    desc_map = _group_desc_map(group_ids)
    return (
        "이번 그룹 항목 정의:\n"
        + _json_dumps(desc_map)
        + "\n이후 evidence 태그 배열이 주어집니다. 각 항목에 대해 긴 문자열로 요약하세요."
    )

# ===== OpenAI 호출 공통 =====
def _chat_json(system: str, user: str) -> Dict[str, Any]:
    """
    response_format={"type":"json_object"} + 메시지에 'json' 포함
    """
    resp = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        return {}

# ===== GitHub payload 구성 =====
def _make_group_payload(gh_data: Dict, group_idx: int) -> Dict[str, Any]:
    """
    github_Fatcher.py 결과를 가정:
      - repo 또는 repo_full_name (있으면)
      - files: List[str]
      - readme: str
      - license_files: Dict[str,str] 또는 List[Dict] 등 유연 처리
      - py_files: Dict[str,str]
    길이 제한을 걸어 과도한 토큰 사용 방지.
    """
    repo = gh_data.get("repo") or gh_data.get("repo_full_name") or gh_data.get("full_name") or ""
    files = gh_data.get("files") or []
    readme = gh_data.get("readme") or ""
    license_files = gh_data.get("license_files") or {}
    py_files = gh_data.get("py_files") or {}

    # 라이선스 파일을 문자열로 합치기(유연 처리)
    if isinstance(license_files, dict):
        lic_text = "\n\n".join([f"# {k}\n{(v or '')[:20000]}" for k, v in list(license_files.items())[:5]])
    elif isinstance(license_files, list):
        # list of dicts or strings
        buf = []
        for item in license_files[:5]:
            if isinstance(item, dict):
                name = item.get("name") or item.get("path") or "LICENSE"
                cont = item.get("content") or ""
                buf.append(f"# {name}\n{cont[:20000]}")
            elif isinstance(item, str):
                buf.append(item[:20000])
        lic_text = "\n\n".join(buf)
    else:
        lic_text = str(license_files)[:20000]

    # py_files 내용 제한
    py_out = {}
    for fn, src in list(py_files.items())[:40]:
        py_out[fn] = (src or "")[:20000]

    # files 목록 제한
    files_out = files[:3000]

    return {
        "repo": repo,
        "files": files_out,
        "readme": readme[:120000],            # README는 길 수 있음
        "license_files": lic_text,
        "py_files": py_out,
    }

def _payload_to_text(payload: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"[repo]\n{payload.get('repo','')}\n")
    parts.append(f"[readme]\n{payload.get('readme','')}\n")
    parts.append(f"[license_files]\n{payload.get('license_files','')}\n")

    # Python 파일
    for fn, code in (payload.get("py_files") or {}).items():
        parts.append(f"[py_files/{fn}]\n{code}\n")

    # 파일 목록
    files = payload.get("files") or []
    if files:
        parts.append("[files]\n" + "\n".join(map(str, files)) + "\n")

    return "\n".join(parts)

# ===== 1단계: evidence 태그 수집 =====
def _recall_collect(group_ids: List[str], payload_text: str) -> Dict[str, List[str]]:
    chunks = _chunk_text(payload_text, CHUNK_CHARS, CHUNK_OVERLAP)
    all_e = {LABELS[k]: [] for k in group_ids}
    for chunk in chunks:
        user = _build_recall_inst(group_ids) + "\n=== PAYLOAD CHUNK ===\n" + chunk
        out = _chat_json(_BASE_RECALL_SYS, user)
        for k in group_ids:
            lbl = LABELS[k]
            vals = out.get(lbl, [])
            if isinstance(vals, list):
                all_e[lbl].extend(vals)
    for lbl in all_e:
        all_e[lbl] = _dedup_list(all_e[lbl], EVIDENCE_LIMIT_PER_KEY)
    return all_e

# ===== 2단계: evidence → 항목별 요약 =====
def _summarize_from_evidence(group_ids: List[str],
                             evidences: Dict[str, List[str]]) -> Dict[str, str]:
    user = _build_summary_inst(group_ids) + "\n=== EVIDENCE TAGS ===\n" + _json_dumps(evidences)
    out = _chat_json(_BASE_SUMMARY_SYS, user)
    result = {}
    for k in group_ids:
        lbl = LABELS[k]
        val = out.get(lbl, "")
        result[lbl] = val if isinstance(val, str) else ""
    return result

# ===== 병합 (sources 제거, evidence 유지) =====
def _merge_for_final(summary_map: Dict[str, str],
                     evidence_map: Dict[str, List[str]]) -> Dict[str, Any]:
    final = {}
    for lbl, txt in summary_map.items():
        final[lbl] = (txt or "").rstrip()
        final[f"{lbl}__evidence"] = evidence_map.get(lbl, [])
    return final

def _merge_dicts(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged

# ===== 외부 인터페이스 (HF 디스패처와 동일한 사용성) =====
def filter_github_features(model_name: str, save: bool = True) -> Dict[str, Any]:
    """
    github_Fatcher.py 가 생성한 github_<org_model>.json 을 입력으로
    16개 항목 기준 evidence 태그를 모아 요약하고, 그룹별 결과를 병합합니다.
    """
    base = model_name.replace("/", "_")
    in_path = f"github_{base}.json"
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"'{in_path}' 파일을 찾을 수 없습니다.")

    with open(in_path, "r", encoding="utf-8") as f:
        gh_data = json.load(f)

    parts = []
    for idx, ids in enumerate(ITEM_GROUPS, start=1):
        try:
            payload = _make_group_payload(gh_data, idx - 1)
            text = _payload_to_text(payload)
            evidences = _recall_collect(ids, text)
            summaries = _summarize_from_evidence(ids, evidences)
            out_grp = _merge_for_final(summaries, evidences)
        except Exception as e:
            print(f"⚠️ 그룹 {idx} 처리 오류: {e}")
            out_grp = {}

        part_path = f"github_filtered_{base}_{idx}.json"
        with open(part_path, "w", encoding="utf-8") as f:
            json.dump(out_grp, f, ensure_ascii=False, indent=2)
        print(f"✅ 그룹 {idx} 결과 저장: {part_path}")
        parts.append(out_grp)

    merged = _merge_dicts(parts)
    if save:
        merged_path = f"github_filtered_{base}.json"
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"✅ 최종 병합 결과 저장: {merged_path}")
    return merged

# ===== CLI =====
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("사용법: python github_Dispatcher.py <org/model>")
        raise SystemExit(1)
    filter_github_features(sys.argv[1])
