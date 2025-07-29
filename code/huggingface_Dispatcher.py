# huggingface_Dispatcher.py
# High-Recall 2-Pass + evidence 태그만 → 요약

import os
import json
import re
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

# ===== 4개 그룹 분할 =====
ITEM_GROUPS: List[List[str]] = [
    ["1-1", "1-2", "1-3", "1-4"],
    ["1-5", "1-6", "2-1", "2-2"],
    ["2-3", "3-1", "3-2", "3-3"],
    ["4-1", "4-2", "4-3", "4-4"],
]

# ===== High-Recall & Model 설정 =====
CHUNK_CHARS = 60_000
CHUNK_OVERLAP = 2_000
EVIDENCE_LIMIT_PER_KEY = 300
MODEL_NAME = os.getenv("OPENAI_MODEL_HF_DISPATCHER", "gpt-4o")
TEMPERATURE = 0

# ===== 정규표현식 및 유틸 =====
_SRC_TAG_RE = re.compile(r'^\s*\[([^\]]+)\]\s*.*$')

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
        x_norm = x.strip()
        if x_norm and x_norm not in seen:
            seen.add(x_norm)
            out.append(x_norm)
            if len(out) >= limit: break
    return out

def _group_desc_map(group_ids: List[str]) -> Dict[str, str]:
    return {LABELS[k]: EVAL_DESCRIPTIONS[LABELS[k]] for k in group_ids}

# ===== 프롬프트 빌더 =====
_BASE_RECALL_SYS = """
당신은 Hugging Face 저장소에서 AI 모델 개방성 평가 정보를 추출하는 전문가입니다.
오직 제공된 payload(원문)만 사용하세요.
결과는 반드시 JSON으로 반환하세요.
"""

_BASE_SUMMARY_SYS = """
당신은 evidence 태그만 사용하여 각 항목별 '길고 상세한' 요약을 작성하는 전문가입니다.
오직 제공된 evidence 태그만 사용하세요.
반드시 json 객체(JSON object)만 반환하세요. 텍스트를 덧붙이지 마세요.
"""

def _build_recall_inst(group_ids: List[str]) -> str:
    desc_map = _group_desc_map(group_ids)
    example = {
        LABELS[k]: [
            "[readme]"
        ] for k in group_ids
    }
    return (
        "이번 그룹 항목 정의:\n"
        + _json_dumps(desc_map)
        + "\n각 키에 대해 evidence 태그만 배열로 반환하세요.\n"
        + "예시 스키마:\n"
        + _json_dumps(example)
    )

_BASE_SUMMARY_SYS = """
당신은 evidence 태그만 사용하여 각 항목별 '길고 상세한' 요약을 작성하는 전문가입니다.
오직 제공된 evidence 태그만 사용하세요.
결과는 반드시 JSON으로 반환하세요.
"""

def _build_summary_inst(group_ids: List[str]) -> str:
    desc_map = _group_desc_map(group_ids)
    return (
        "이번 그룹 항목 정의:\n"
        + _json_dumps(desc_map)
        + "\n이후 evidence 태그 배열이 주어집니다. 각 항목에 대해 긴 문자열로 요약하세요."
    )

# ===== GPT 호출 =====
def _chat_json(system: str, user: str) -> Dict[str, Any]:
    resp = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user}
        ],
        temperature=TEMPERATURE,
        response_format={"type": "json_object"}    # ← 문자열 대신 객체로 지정
    )
    text = resp.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except:
        return {}

# ===== evidence 수집 =====
def _make_group_payload(hf_data: Dict, group_idx: int) -> Dict:
    py_src = hf_data.get("py_files", {}) or {}
    py_items = list(py_src.items())[:20]
    py_files = {fn: (src[:20_000] if isinstance(src, str) else "") for fn, src in py_items}
    return {
        "model_id": hf_data.get("model_id", ""),
        "files": hf_data.get("files", [])[:2000],
        "readme": hf_data.get("readme", ""),
        "license_file": hf_data.get("license_file", ""),
        "config": hf_data.get("config", ""),
        "generation_config": hf_data.get("generation_config", ""),
        "py_files": py_files,
    }

def _payload_to_text(payload: Dict) -> str:
    parts = [
        f"[model_id]\n{payload.get('model_id','')}\n",
        f"[readme]\n{payload.get('readme','')}\n",
        f"[license_file]\n{payload.get('license_file','')}\n",
        f"[config]\n{payload.get('config','')}\n",
        f"[generation_config]\n{payload.get('generation_config','')}\n",
    ]
    for fn, code in payload.get("py_files", {}).items():
        parts.append(f"[py_files/{fn}]\n{code}\n")
    files = payload.get("files", [])
    if files:
        parts.append("[files]\n" + "\n".join(map(str, files)) + "\n")
    return "\n".join(parts)

def _recall_collect(group_ids: List[str], payload_text: str) -> Dict[str, List[str]]:
    chunks = _chunk_text(payload_text, CHUNK_CHARS, CHUNK_OVERLAP)
    all_e = {LABELS[k]: [] for k in group_ids}
    for chunk in chunks:
        out = _chat_json(_BASE_RECALL_SYS, _build_recall_inst(group_ids) + "\n=== PAYLOAD ===\n" + chunk)
        for k in group_ids:
            vals = out.get(LABELS[k], [])
            if isinstance(vals, list):
                all_e[LABELS[k]].extend(vals)
    for lbl in all_e:
        all_e[lbl] = _dedup_list(all_e[lbl], EVIDENCE_LIMIT_PER_KEY)
    return all_e

def _summarize_from_evidence(group_ids: List[str], evidences: Dict[str, List[str]]) -> Dict[str, str]:
    user = _build_summary_inst(group_ids) + "\n=== EVIDENCE ===\n" + _json_dumps(evidences)
    out = _chat_json(_BASE_SUMMARY_SYS, user)
    return {LABELS[k]: out.get(LABELS[k], "") for k in group_ids}

def _merge_for_final(summary_map: Dict[str, str],
                     evidence_map: Dict[str, List[str]]) -> Dict[str, Any]:
    final = {}
    for lbl, txt in summary_map.items():
        final[lbl] = txt.rstrip()
        final[f"{lbl}__evidence"] = evidence_map.get(lbl, [])
    return final

def _merge_dicts(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged

def filter_hf_features(model_name: str, save: bool = True) -> Dict[str, Any]:
    base = model_name.replace("/", "_").lower()
    path_in = f"huggingface_{base}.json"
    if not os.path.exists(path_in):
        raise FileNotFoundError(f"'{path_in}' 파일이 없습니다.")
    hf_data = json.load(open(path_in, encoding="utf-8"))

    all_parts = []
    for idx, grp in enumerate(ITEM_GROUPS, start=1):
        try:
            payload = _make_group_payload(hf_data, idx-1)
            text = _payload_to_text(payload)
            evidences = _recall_collect(grp, text)
            summaries = _summarize_from_evidence(grp, evidences)
            part = _merge_for_final(summaries, evidences)
        except Exception as e:
            print(f"⚠️ 그룹 {idx} 처리 오류: {e}")
            part = {}
        out_path = f"huggingface_filtered_{base}_{idx}.json"
        json.dump(part, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"✅ 그룹 {idx} 결과 저장: {out_path}")
        all_parts.append(part)

    merged = _merge_dicts(all_parts)
    if save:
        out_merged = f"huggingface_filtered_{base}.json"
        json.dump(merged, open(out_merged, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"✅ 최종 병합 결과 저장: {out_merged}")
    return merged

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("사용법: python huggingface_Dispatcher.py <org/model>")
        sys.exit(1)
    filter_hf_features(sys.argv[1])
