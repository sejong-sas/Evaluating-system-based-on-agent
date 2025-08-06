# arxiv_Dispatcher.py
# High-Recall 2-Pass  (evidence {source, quote} → 긴 요약)
# - evidence를 객체 배열로 저장
# - 요약은 quote 들만 사용
# - __evidence_sources / __sources 제거
# - 입력: arxiv_fulltext_{base}.json → 없으면 arxiv_{base}.json

import os, json, re
from typing import Dict, List, Any
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# ─────────────────── 환경 ───────────────────
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
if not key:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
_client = OpenAI(api_key=key)

# ─────────── 16개 항목 라벨 ───────────
LABELS = {
    "1-1": "1-1 (가중치 Weights)",           "1-2": "1-2 (코드 Code)",
    "1-3": "1-3 (라이선스 License)",         "1-4": "1-4 (논문 Paper)",
    "1-5": "1-5 (아키텍처 Architecture)",    "1-6": "1-6 (토크나이저 Tokenizer)",
    "2-1": "2-1 (하드웨어 Hardware)",         "2-2": "2-2 (소프트웨어 Software)",
    "2-3": "2-3 (API)",
    "3-1": "3-1 (사전학습 Pre-training)",     "3-2": "3-2 (파인튜닝 Fine-tuning)",
    "3-3": "3-3 (강화학습 Reinforcement Learning)",
    "4-1": "4-1 (사전학습 데이터 Pre-training Data)",
    "4-2": "4-2 (파인튜닝 데이터 Fine-tuning Data)",
    "4-3": "4-3 (강화학습 데이터 Reinforcement Learning Data)",
    "4-4": "4-4 (데이터 필터링 Data Filtering)",
}

EVAL_DESCRIPTIONS = {
    LABELS["1-1"]: "모델 가중치의 공개 여부, 위치, 접근 방식, 누구나 다운로드 가능한지에 관련된 모든 내용",
    LABELS["1-2"]: "모델 훈련 및 실행을 위한 코드가 공개되었는지, 어떤 부분이 공개되었는지에 관련된 모든 내용",
    LABELS["1-3"]: "라이선스의 존재 여부, 종류, 허용된 권리(사용, 수정, 배포, 상업적 이용)에 관련된 모든 내용",
    LABELS["1-4"]: "모델과 관련된 공식 논문, 기술 보고서, 블로그 등 문서의 존재와 링크에 관련된 모든 내용",
    LABELS["1-5"]: "모델 아키텍처(레이어 수, 하이퍼파라미터 등)와 구조 설계의 세부 정보에 관련된 모든 내용",
    LABELS["1-6"]: "어떤 토크나이저를 사용하는지, 이름과 구조, 다운로드 가능 여부에 관련된 모든 내용",
    LABELS["2-1"]: "모델 훈련에 사용된 하드웨어 종류(H100, TPU 등), 수량, 계산 자원 규모에 관련된 모든 내용",
    LABELS["2-2"]: "훈련에 사용된 소프트웨어(프레임워크, 라이브러리 등)의 종류, 버전, 설정에 관련된 모든 내용",
    LABELS["2-3"]: "모델이 접근 가능한 API(gpt api, gemini api 같은 api여야 함 라이브러리x)의 존재 여부, 문서 링크, 사용 예제, 공개 여부에 관련된 모든 내용",
    LABELS["3-1"]: "사전학습 시 사용된 방법론, 절차, 데이터 흐름, 하이퍼파라미터 설정 등에 관련된 모든 내용",
    LABELS["3-2"]: "파인튜닝 방식, 목적, 데이터 사용 여부, 재현 가능한 파이프라인 존재 여부에 관련된 모든 내용",
    LABELS["3-3"]: "RLHF, DPO 등 강화학습 알고리즘 사용 여부, 구체적인 방식과 절차, 설정값 등에 관련된 모든 내용",
    LABELS["4-1"]: "사전학습에 사용된 데이터의 종류, 수량, 출처, 사용 범위 및 구성 방식에 관련된 모든 내용",
    LABELS["4-2"]: "파인튜닝에 사용된 데이터셋의 출처, 구성, 데이터 예시, 공개 여부 등에 관련된 모든 내용",
    LABELS["4-3"]: "강화학습에 사용된 데이터셋의 구성, 접근 가능 여부, 출처, 생성 방식에 관련된 모든 내용",
    LABELS["4-4"]: "데이터 필터링 또는 정제 방법, 사용된 기준, 필터링 과정과 그 영향에 관련된 모든 내용",
}

# ─────────── 4개 그룹 ───────────
ITEM_GROUPS = [
    ["1-1","1-2","1-3","1-4"],
    ["1-5","1-6","2-1","2-2"],
    ["2-3","3-1","3-2","3-3"],
    ["4-1","4-2","4-3","4-4"],
]

# ─────────── 파라미터 ───────────
CHUNK_CHARS = 60_000
CHUNK_OVERLAP = 2_000
EVIDENCE_LIMIT_PER_KEY = 300
MODEL_NAME = os.getenv("OPENAI_MODEL_ARXIV_DISPATCHER", "o3-mini")

# ─────────── 유틸 ───────────
def _js(o): return json.dumps(o, ensure_ascii=False, indent=2)

def _chunk(s, n, ov):
    out, L, i = [], len(s), 0
    while i < L:
        end = min(i + n, L)
        out.append(s[i:end])
        if end == L: break
        i = end - ov if end - ov > i else end
    return out

def _dedup_evs(evs: List[Dict[str,str]], limit:int):
    seen, out = set(), []
    for ev in evs:
        if not (isinstance(ev, dict)
                and isinstance(ev.get("source"), str)
                and isinstance(ev.get("quote"), str)):
            continue
        src, qt = ev["source"].strip(), ev["quote"].strip()
        if not src or not qt: continue
        key = (src, qt)
        if key in seen: continue
        seen.add(key); out.append({"source": src, "quote": qt})
        if len(out) >= limit: break
    return out

# ─────────── 프롬프트 ───────────
_BASE_RECALL_SYS = """
당신은 arXiv 원문에서 AI 모델 개방성 평가 정보를 추출하는 전문가입니다.
payload(원문)만 사용하여 각 항목 evidence를
  [{ "source": "...", "quote": "..." }, …]  형식으로 반환하십시오.
· source : [title], [abstract], [pdf_text], [sections/Introduction] … 등
· quote  : 해당 섹션에서 그대로 복사한 문장
근거가 없으면 빈 배열 [].
반드시 JSON 객체만 출력하십시오.
""".strip()

_BASE_SUMMARY_SYS = """
주어진 quote 만 사용하여 각 항목을 길고 자세히 요약하십시오.
반드시 JSON 객체만 출력하십시오.
""".strip()

def _desc(ids): return {LABELS[i]: EVAL_DESCRIPTIONS[LABELS[i]] for i in ids}          # 설명 생략(토큰 절약)
def _recall_inst(g): return "이번 그룹 항목:\n"+_js(_desc(g))
def _summ_inst(g):   return "이번 그룹 항목:\n"+_js(_desc(g))

# ─────────── GPT JSON 호출 ───────────
def _chat_json(sys, usr):
    r = _client.chat.completions.create(
        model=MODEL_NAME, reasoning_effort="medium",
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":usr}]
    )
    try:    return json.loads(r.choices[0].message.content.strip())
    except: return {}

# ─────────── payload 빌드 ───────────
def _make_payload(d):
    # 기본 필드
    pid   = d.get("arxiv_id") or d.get("id") or ""
    title = d.get("title") or d.get("meta",{}).get("title") or ""
    abstr = d.get("abstract") or ""
    cats  = d.get("categories") or ""
    lic   = d.get("license") or ""
    auth  = d.get("authors")  or ""
    body  = d.get("pdf_text") or d.get("fulltext") or d.get("body") or ""

    # section 텍스트
    secs = []
    for s in (d.get("sections") or [])[:50]:
        if isinstance(s, dict):
            secs.append({"title": str(s.get("title",""))[:300],
                         "text":  str(s.get("text",""))[:20_000]})
    if not secs and isinstance(d.get("section_texts"), dict):
        for k,v in list(d["section_texts"].items())[:50]:
            secs.append({"title": str(k)[:300], "text": str(v)[:20_000]})

    # bibliography
    bib = d.get("references") or d.get("bib") or ""
    if isinstance(bib, list):
        bib = "\n".join(str(x) for x in bib[:3000])
    else:
        bib = str(bib)[:100_000]

    return {
        "arxiv_id": pid,
        "title":    str(title)[:2000],
        "abstract": str(abstr)[:120_000],
        "categories": str(cats)[:5000],
        "license": str(lic)[:5000],
        "authors": str(auth)[:8000],
        "pdf_text": str(body)[:240_000],
        "sections": secs,
        "bib": bib,
    }

def _payload_text(p):
    parts = [
        f"[arxiv_id]\n{p['arxiv_id']}\n",
        f"[title]\n{p['title']}\n",
        f"[authors]\n{p['authors']}\n",
        f"[categories]\n{p['categories']}\n",
        f"[license]\n{p['license']}\n",
        f"[abstract]\n{p['abstract']}\n",
        f"[pdf_text]\n{p['pdf_text']}\n"
    ]
    for s in p["sections"]:
        parts.append(f"[sections/{s['title'] or 'Section'}]\n{s['text']}\n")
    if p["bib"]:
        parts.append("[bib]\n"+p["bib"]+"\n")
    return "\n".join(parts)

# ─────────── evidence 수집 ───────────
_ALLOWED = ("arxiv_id","title","abstract","pdf_text","sections/",
            "categories","license","authors","bib")

def _valid(src): return isinstance(src,str) and src.startswith(_ALLOWED)

def _collect(g, text):
    ev = {LABELS[k]: [] for k in g}
    for ch in _chunk(text, CHUNK_CHARS, CHUNK_OVERLAP):
        ans = _chat_json(_BASE_RECALL_SYS, _recall_inst(g)+"\n=== PAYLOAD ===\n"+ch)
        for k in g:
            ev[LABELS[k]].extend(ans.get(LABELS[k], []))
    for k in ev:
        ev[k] = _dedup_evs(ev[k], EVIDENCE_LIMIT_PER_KEY)
    return ev

# ─────────── 요약 ───────────
def _summarize(g, ev):
    quotes = {LABELS[k]: [e["quote"] for e in ev[LABELS[k]]] for k in g}
    ans = _chat_json(_BASE_SUMMARY_SYS, _summ_inst(g)+"\n=== QUOTES ===\n"+_js(quotes))
    return {LABELS[k]: ans.get(LABELS[k], "") for k in g}

# ─────────── 병합 ───────────
def _merge(sum_, ev):
    return {lbl: sum_[lbl].strip() for lbl in sum_} | {
        f"{lbl}__evidence": ev.get(lbl, []) for lbl in sum_
    }

def _merge_all(lst):
    m={}
    for d in lst: m.update(d)
    return m

# ─────────── 외부 함수 ───────────
def filter_arxiv_features(model, save: bool = True, output_dir: str | Path = "."):
    base = model.replace("/", "_").lower()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 입력: outdir 우선 → 루트 폴백
    candidates = [
        output_dir / f"arxiv_fulltext_{base}.json",
        output_dir / f"arxiv_{base}.json",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if not src:
        alt = next(
            (Path(p) for p in [f"arxiv_fulltext_{base}.json", f"arxiv_{base}.json"] if Path(p).exists()),
            None
        )
        if not alt:
            raise FileNotFoundError([str(c) for c in candidates])
        src = alt

    ax = json.load(open(src, encoding="utf-8"))

    # 2) 페처 스키마 정규화: {"full_texts":[{"arxiv_id","full_text"}...]} → 단일 문서 형태
    doc_for_payload = ax
    if isinstance(ax, dict) and "full_texts" in ax:
        texts, ids = [], []
        for t in ax.get("full_texts", []):
            if not isinstance(t, dict):
                continue
            ids.append(str(t.get("arxiv_id", "")).strip())
            # 다양한 키 폴백: full_text 우선, 없으면 pdf_text
            texts.append(str(t.get("full_text") or t.get("pdf_text") or ""))

        merged_text = "\n\n".join([x for x in texts if x])
        merged_id = ";".join([x for x in ids if x])

        doc_for_payload = {
            "arxiv_id": merged_id,
            "title": "",
            "abstract": "",
            "categories": "",
            "license": "",
            "authors": "",
            # ★ 핵심: 병합 텍스트를 pdf_text로 매핑 (디스패처의 기대 스키마에 맞춤)
            "pdf_text": merged_text,
            "sections": [],
            "bib": "",
        }

    # 3) 그룹별 처리
    parts = []
    for i, grp in enumerate(ITEM_GROUPS, 1):
        try:
            pay = _make_payload(doc_for_payload)
            text = _payload_text(pay)
            ev = _collect(grp, text)
            summ = _summarize(grp, ev)
            part = _merge(summ, ev)
        except Exception as e:
            print(f"⚠️ 그룹 {i} 오류:", e)
            part = {}

        if save:
            fp = output_dir / f"arxiv_filtered_{base}_{i}.json"
            json.dump(part, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print("✅ 그룹", i, "저장:", fp)
        parts.append(part)

    # 4) 최종 병합 저장
    merged = _merge_all(parts)
    if save:
        fp = output_dir / f"arxiv_filtered_final_{base}.json"
        json.dump(merged, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("✅ 최종 병합 저장:", fp)
    return merged


# ─────────── CLI ───────────
if __name__=="__main__":
    import sys
    mid="bigscience/bloomz-560m"
    if len(sys.argv)>1 and sys.argv[1]: mid=sys.argv[1]
    print("▶ 실행 모델:", mid)
    filter_arxiv_features(mid)