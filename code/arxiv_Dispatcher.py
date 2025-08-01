# arxiv_Dispatcher.py
# High-Recall 2-Pass (evidence 태그만 수집 → 요약) / 4개 그룹 구조
# - OpenAI Chat Completions 사용
# - response_format={"type":"json_object"} 강제
# - 시스템/유저 메시지에 'json' 명시
# - __sources 제거, __evidence(태그 배열)만 유지
# - 입력 파일명 호환: arxiv_fulltext_{base}.json 또는 arxiv_{base}.json 모두 지원

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

# ===== 4개 그룹 =====
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
MODEL_NAME = os.getenv("OPENAI_MODEL_ARXIV_DISPATCHER", "o3-mini")
#TEMPERATURE = 0

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

# ===== 프롬프트 (json 단어 포함) =====
_BASE_RECALL_SYS = """
당신은 arXiv 원문에서 AI 모델 개방성 평가 정보를 추출하는 전문가입니다.
오직 제공된 payload(원문)만 사용하세요. 외부 자료 금지.
만약 증거가 있는경우에는 그 근거 문장을 포함 시켜 주고, 없는 경우에는 "증거 없음"이라고 명시하세요.
반드시 json 객체(JSON object)만 반환하세요. 텍스트/설명/백틱을 덧붙이지 마세요.
"""

def _build_recall_inst(group_ids: List[str]) -> str:
    """
    evidence 태그만 배열로 수집.
    허용 태그 예시:
      [arxiv_id], [title], [authors], [abstract], [categories], [license],
      [pdf_text], [sections/Introduction], [sections/Methods], [sections/Experiments],
      [sections/Results], [sections/Discussion], [sections/Conclusion], [bib]
    """
    desc_map = _group_desc_map(group_ids)
    example = {
        LABELS[k]: ["[title]", "[abstract]", "[pdf_text]", "[sections/Introduction]", "[bib]"]
        for k in group_ids
    }
    return (
        "이번 그룹 항목 정의:\n"
        + _json_dumps(desc_map)
        + "\n각 키에 대해 evidence '태그'만 배열로 반환하세요.\n"
        + "허용 태그 예: [title], [abstract], [pdf_text], [sections/Methods], [bib], [license]\n"
        + "예시 스키마:\n"
        + _json_dumps(example)
    )

_BASE_SUMMARY_SYS = """
당신은 arXiv evidence 태그만 사용해 각 항목별 '길고 상세한' 요약을 작성하는 전문가입니다.
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

# ===== OpenAI 호출 =====
def _chat_json(system: str, user: str) -> Dict[str, Any]:
    resp = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        reasoning_effort="medium",
        #temperature=TEMPERATURE,
        response_format={"type": "json_object"},  # 반드시 객체
    )
    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        return {}

# ===== payload 구성 =====
def _make_group_payload(ax_data: Dict[str, Any], group_idx: int) -> Dict[str, Any]:
    """
    arxiv_Fetcher가 만드는 JSON의 다양한 키를 유연하게 수용:
      - arxiv_id / id
      - title / meta.title
      - abstract
      - categories
      - license
      - authors
      - pdf_text / fulltext
      - sections (list of {title, text}) / section_texts (dict)
      - references / bib
    """
    # 기본 키
    arxiv_id = ax_data.get("arxiv_id") or ax_data.get("id") or ""
    title = ax_data.get("title") or ax_data.get("meta", {}).get("title") or ""
    abstract = ax_data.get("abstract") or ""
    categories = ax_data.get("categories") or []
    license_ = ax_data.get("license") or ""
    authors = ax_data.get("authors") or []
    # 본문 텍스트는 다양한 필드명을 허용
    pdf_text = ax_data.get("pdf_text") or ax_data.get("fulltext") or ax_data.get("body") or ""

    # 섹션: list 형태 우선, 다음으로 dict 형태를 펼침
    sections = ax_data.get("sections") or []
    section_texts = ax_data.get("section_texts") or {}

    norm_sections: List[Dict[str, str]] = []
    if isinstance(sections, list) and sections:
        for s in sections[:50]:
            if isinstance(s, dict):
                stitle = str(s.get("title", ""))[:300]
                stext = str(s.get("text", ""))[:20000]
                if stitle or stext:
                    norm_sections.append({"title": stitle, "text": stext})
    elif isinstance(section_texts, dict) and section_texts:
        # { "Introduction": "...", "Methods": "..." } 형태
        for stitle, stext in list(section_texts.items())[:50]:
            norm_sections.append({"title": str(stitle)[:300], "text": str(stext)[:20000]})

    # 참고문헌
    references = ax_data.get("references") or ax_data.get("bib") or ""
    if isinstance(references, list):
        if all(isinstance(x, str) for x in references):
            bib_text = "\n".join(references[:3000])
        else:
            buf = []
            for r in references[:1000]:
                if isinstance(r, dict):
                    t = r.get("title") or r.get("raw") or ""
                    buf.append(str(t))
            bib_text = "\n".join(buf)
    else:
        bib_text = str(references)[:100000]

    # 문자열 정규화
    if isinstance(categories, list):
        categories_str = ", ".join([str(x) for x in categories])[:5000]
    else:
        categories_str = str(categories)[:5000]

    if isinstance(authors, list):
        authors_str = ", ".join([str(x) for x in authors])[:8000]
    else:
        authors_str = str(authors)[:8000]

    return {
        "arxiv_id": arxiv_id,
        "title": str(title)[:2000],
        "abstract": str(abstract)[:120000],
        "categories": categories_str,
        "authors": authors_str,
        "license": str(license_)[:5000],
        "pdf_text": str(pdf_text)[:240000],
        "sections": norm_sections,   # [{title, text}]
        "bib": bib_text,
    }

def _payload_to_text(payload: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(f"[arxiv_id]\n{payload.get('arxiv_id','')}\n")
    parts.append(f"[title]\n{payload.get('title','')}\n")
    parts.append(f"[authors]\n{payload.get('authors','')}\n")
    parts.append(f"[categories]\n{payload.get('categories','')}\n")
    parts.append(f"[license]\n{payload.get('license','')}\n")
    parts.append(f"[abstract]\n{payload.get('abstract','')}\n")
    parts.append(f"[pdf_text]\n{payload.get('pdf_text','')}\n")
    for s in payload.get("sections", []):
        stitle = s.get("title", "") or "Section"
        stext = s.get("text", "")
        parts.append(f"[sections/{stitle}]\n{stext}\n")
    bib = payload.get("bib", "")
    if bib:
        parts.append(f"[bib]\n{bib}\n")
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

# ===== 외부 인터페이스 =====
def filter_arxiv_features(model_name: str, save: bool = True) -> Dict[str, Any]:
    """
    arxiv_Fetcher.py 가 생성한 JSON을 입력으로
    16개 항목 기준 evidence 태그를 모아 요약하고, 그룹별 결과를 병합합니다.

    입력 파일명 후보:
      - arxiv_fulltext_{base}.json  (우선)
      - arxiv_{base}.json           (백업 호환)
    """
    base = model_name.replace("/", "_").lower()
    candidates = [
        f"arxiv_fulltext_{base}.json",
        f"arxiv_{base}.json",
    ]

    in_path = None
    for p in candidates:
        if os.path.exists(p):
            in_path = p
            break
    if not in_path:
        raise FileNotFoundError(f"arXiv JSON 파일을 찾지 못했습니다. 시도한 경로: {candidates}")

    with open(in_path, "r", encoding="utf-8") as f:
        ax_data = json.load(f)

    parts = []
    for idx, ids in enumerate(ITEM_GROUPS, start=1):
        try:
            payload = _make_group_payload(ax_data, idx - 1)
            text = _payload_to_text(payload)
            evidences = _recall_collect(ids, text)
            summaries = _summarize_from_evidence(ids, evidences)
            out_grp = _merge_for_final(summaries, evidences)
        except Exception as e:
            print(f"⚠️ 그룹 {idx} 처리 오류: {e}")
            out_grp = {}

        part_path = f"arxiv_filtered_{base}_{idx}.json"
        with open(part_path, "w", encoding="utf-8") as f:
            json.dump(out_grp, f, ensure_ascii=False, indent=2)
        print(f"✅ 그룹 {idx} 결과 저장: {part_path}")
        parts.append(out_grp)

    merged = _merge_dicts(parts)
    if save:
        merged_path = f"arxiv_filtered_final_{base}.json"
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"✅ 최종 병합 결과 저장: {merged_path}")
    return merged

# ===== CLI =====
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("사용법: python arxiv_Dispatcher.py <org/model>")
        raise SystemExit(1)
    filter_arxiv_features(sys.argv[1])