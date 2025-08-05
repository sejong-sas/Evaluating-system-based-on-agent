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

def _desc(ids): return {LABELS[i]: "" for i in ids}          # 설명 생략(토큰 절약)
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
def filter_arxiv_features(model, save=True):
    base = model.replace("/","_").lower()
    paths = [f"arxiv_fulltext_{base}.json", f"arxiv_{base}.json"]
    src = next((p for p in paths if os.path.exists(p)), None)
    if not src:
        raise FileNotFoundError(paths)
    ax = json.load(open(src,encoding="utf-8"))

    parts=[]
    for i,grp in enumerate(ITEM_GROUPS,1):
        try:
            pay  = _make_payload(ax)
            text = _payload_text(pay)
            ev   = _collect(grp, text)
            summ = _summarize(grp, ev)
            part = _merge(summ, ev)
        except Exception as e:
            print(f"⚠️ 그룹 {i} 오류:", e); part={}
        if save:
            fp=f"arxiv_filtered_{base}_{i}.json"
            json.dump(part, open(fp,"w",encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print("✅ 그룹", i, "저장:", fp)
        parts.append(part)

    merged=_merge_all(parts)
    if save:
        fp=f"arxiv_filtered_final_{base}.json"
        json.dump(merged, open(fp,"w",encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("✅ 최종 병합 저장:", fp)
    return merged

# ─────────── CLI ───────────
if __name__=="__main__":
    import sys
    mid="bigscience/bloomz-560m"
    if len(sys.argv)>1 and sys.argv[1]: mid=sys.argv[1]
    print("▶ 실행 모델:", mid)
    filter_arxiv_features(mid)
