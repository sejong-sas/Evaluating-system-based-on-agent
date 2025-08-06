# github_Dispatcher.py
# High-Recall 2-Pass  (evidence {source, quote} → 긴 요약)
# - evidence를 객체 배열 [{source, quote}, …] 로 저장
# - 요약은 quote 들만 사용
# - __evidence_sources, __sources  등 불필요 필드 제거

import os, json, re
from typing import Dict, List, Any
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# ────────────────── 환경 ──────────────────
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
_client = OpenAI(api_key=_api_key)

# ─────────────── 16개 평가 항목 ───────────────
LABELS = {
    "1-1": "1-1 (가중치 Weights)",               "1-2": "1-2 (코드 Code)",
    "1-3": "1-3 (라이선스 License)",            "1-4": "1-4 (논문 Paper)",
    "1-5": "1-5 (아키텍처 Architecture)",        "1-6": "1-6 (토크나이저 Tokenizer)",
    "2-1": "2-1 (하드웨어 Hardware)",            "2-2": "2-2 (소프트웨어 Software)",
    "2-3": "2-3 (API)",
    "3-1": "3-1 (사전학습 Pre-training)",        "3-2": "3-2 (파인튜닝 Fine-tuning)",
    "3-3": "3-3 (강화학습 Reinforcement Learning)",
    "4-1": "4-1 (사전학습 데이터 Pre-training Data)",
    "4-2": "4-2 (파인튜닝 데이터 Fine-tuning Data)",
    "4-3": "4-3 (강화학습 데이터 Reinforcement Learning Data)",
    "4-4": "4-4 (데이터 필터링 Data Filtering)",
}

# ─────────────── 항목 설명 (간략) ───────────────
EVAL_DESCRIPTIONS = {
    LABELS["1-1"]: "가중치 공개 여부·위치·접근 방식",
    LABELS["1-2"]: "훈련/추론 코드 공개 여부·범위",
    LABELS["1-3"]: "라이선스 종류·권한",
    LABELS["1-4"]: "공식 논문/보고서·링크",
    LABELS["1-5"]: "모델 아키텍처 상세",
    LABELS["1-6"]: "토크나이저 종류·공개 여부",
    LABELS["2-1"]: "훈련 하드웨어 종류·규모",
    LABELS["2-2"]: "사용 SW/프레임워크·버전",
    LABELS["2-3"]: "모델 API 존재·문서·사용 예",
    LABELS["3-1"]: "사전학습 방법·하이퍼파라미터",
    LABELS["3-2"]: "파인튜닝 방법·파이프라인",
    LABELS["3-3"]: "RLHF/DPO 등 강화학습 방법",
    LABELS["4-1"]: "사전학습 데이터 종류·규모·출처",
    LABELS["4-2"]: "파인튜닝 데이터 출처·구성·공개 여부",
    LABELS["4-3"]: "강화학습 데이터 구성·출처",
    LABELS["4-4"]: "데이터 필터링 기준·과정",
}

# ───────────── 그룹 ─────────────
ITEM_GROUPS = [
    ["1-1", "1-2", "1-3", "1-4"],
    ["1-5", "1-6", "2-1", "2-2"],
    ["2-3", "3-1", "3-2", "3-3"],
    ["4-1", "4-2", "4-3", "4-4"],
]

# ───────────── 파라미터 ─────────────
CHUNK_CHARS = 60_000
CHUNK_OVERLAP = 2_000
EVIDENCE_LIMIT_PER_KEY = 300
MODEL_NAME = os.getenv("OPENAI_MODEL_GH_DISPATCHER", "o3-mini")

# ───────────── 유틸 ─────────────
def _js(o: Any) -> str: return json.dumps(o, ensure_ascii=False, indent=2)

def _chunk(s: str, size: int, ov: int) -> List[str]:
    out, n, i = [], len(s), 0
    while i < n:
        end = min(i + size, n)
        out.append(s[i:end])
        if end == n: break
        i = end - ov if end - ov > i else end
    return out

def _dedup_evid(evs: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
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

def _desc(ids: List[str]): return {LABELS[i]: EVAL_DESCRIPTIONS[LABELS[i]] for i in ids}

# ───────────── 프롬프트 ─────────────
_BASE_RECALL_SYS = """
당신은 GitHub 저장소에서 AI 모델 개방성 평가 정보를 추출하는 전문가입니다.
payload(원문)만 사용하여, 각 항목별 evidence를
  [{ "source": "...", "quote": "..." }, …]  형식으로 반환하십시오.
· source  : [repo], [readme], [license_files], [files], [py_files/xxx.py] 중 하나
· quote   : 해당 섹션에서 그대로 복사한 문장(수정·요약 금지)
근거가 없으면 빈 배열 [].
반드시 JSON 객체만 출력하십시오.
""".strip()

_BASE_SUMMARY_SYS = """
주어진 quote 만 사용하여 각 항목을 길고 자세히 요약하십시오.
반드시 JSON 객체만 출력하십시오.
""".strip()

def _recall_inst(g: List[str]) -> str:
    return (
        f"이번 그룹 항목 정의:\n{_js(_desc(g))}\n"
        "예시 스키마(형식 참고):\n" +
        _js({LABELS[k]: [{"source": "readme", "quote": "문장"}] for k in g})
    )

def _summary_inst(g: List[str]) -> str:
    return f"이번 그룹 항목 정의:\n{_js(_desc(g))}\nquote 배열이 제공됩니다. 항목별로 요약하십시오."

# ───────────── GPT ↔ JSON ─────────────
def _chat_json(sys_msg: str, usr: str) -> Dict[str, Any]:
    r = _client.chat.completions.create(
        model=MODEL_NAME, reasoning_effort="medium",
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":sys_msg},
                  {"role":"user","content":usr}]
    )
    try:    return json.loads(r.choices[0].message.content.strip())
    except: return {}

# ───────────── payload 빌드 ─────────────
def _make_payload(d: Dict, _: int) -> Dict:
    repo  = d.get("repo") or d.get("full_name") or ""
    files = (d.get("files") or [])[:3000]
    readme = (d.get("readme") or "")[:120_000]

    # license
    lic = d.get("license_files") or {}
    if isinstance(lic, dict):
        lic_text = "\n\n".join(
            f"# {k}\n{(v or '')[:20_000]}" for k, v in list(lic.items())[:5]
        )
    elif isinstance(lic, list):
        buf = []
        for it in lic[:5]:
            if isinstance(it, dict):
                name = it.get("name") or it.get("path") or "LICENSE"
                buf.append(f"# {name}\n{(it.get('content') or '')[:20_000]}")
            elif isinstance(it, str):
                buf.append(it[:20_000])
        lic_text = "\n\n".join(buf)
    else:
        lic_text = str(lic)[:20_000]

    py_files = {}
    for fn, src in (d.get("py_files") or {}).items():
        if len(py_files) >= 40: break
        py_files[fn] = (src or "")[:20_000]

    return {"repo": repo, "files": files, "readme": readme,
            "license_files": lic_text, "py_files": py_files}

def _payload_text(p: Dict) -> str:
    parts = [
        f"[repo]\n{p['repo']}\n",
        f"[readme]\n{p['readme']}\n",
        f"[license_files]\n{p['license_files']}\n",
    ]
    for fn, code in p["py_files"].items():
        parts.append(f"[py_files/{fn}]\n{code}\n")
    if p["files"]:
        parts.append("[files]\n" + "\n".join(p["files"]) + "\n")
    return "\n".join(parts)

# ───────────── 단계 1: evidence 수집 ─────────────
_ALLOWED = ("repo", "readme", "license_files", "files", "py_files/")

def _valid(src: str): return isinstance(src, str) and src.startswith(_ALLOWED)

def _collect(g: List[str], text: str) -> Dict[str, List[Dict[str, str]]]:
    ev = {LABELS[k]: [] for k in g}
    for ch in _chunk(text, CHUNK_CHARS, CHUNK_OVERLAP):
        ans = _chat_json(_BASE_RECALL_SYS, _recall_inst(g) + "\n=== PAYLOAD ===\n" + ch)
        for k in g:
            arr = ans.get(LABELS[k], [])
            if isinstance(arr, list):
                ev[LABELS[k]].extend(arr)
    for k in ev:
        ev[k] = _dedup_evid(ev[k], EVIDENCE_LIMIT_PER_KEY)
    return ev

# ───────────── 단계 2: 요약 ─────────────
def _summarize(g: List[str], ev: Dict[str, List[Dict[str, str]]]) -> Dict[str, str]:
    quotes = {LABELS[k]: [e["quote"] for e in ev[LABELS[k]]] for k in g}
    ans = _chat_json(_BASE_SUMMARY_SYS, _summary_inst(g) +
                     "\n=== QUOTES ===\n" + _js(quotes))
    return {LABELS[k]: ans.get(LABELS[k], "") for k in g}

# ───────────── 병합 ─────────────
def _merge(summary: Dict[str, str], ev: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    return {lbl: summary[lbl].strip() for lbl in summary} | {
        f"{lbl}__evidence": ev.get(lbl, []) for lbl in summary
    }

def _merge_all(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
    m = {}
    for d in lst: m.update(d)
    return m

# ───────────── 외부 함수 ─────────────
def filter_github_features(model: str, save: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    base = model.replace("/", "_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"github_{base}.json"              # ★ 입력 우선 outdir
    if not path.exists():
        alt = Path(f"github_{base}.json")                  # 루트 폴백
        if not alt.exists():
            raise FileNotFoundError(str(path))
        gh = json.load(open(alt, encoding="utf-8"))
    else:
        gh = json.load(open(path, encoding="utf-8"))

    parts = []
    for idx, grp in enumerate(ITEM_GROUPS, 1):
        try:
            payload = _make_payload(gh, idx-1)
            text = _payload_text(payload)
            ev    = _collect(grp, text)
            summ  = _summarize(grp, ev)
            part  = _merge(summ, ev)
        except Exception as e:
            print(f"⚠️ 그룹 {idx} 처리 오류:", e)
            part = {}
        out = output_dir / f"github_filtered_{base}_{idx}.json"     # ★ 저장 outdir
        if save:
            json.dump(part, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
            print("✅ 그룹", idx, "결과 저장:", out)
        parts.append(part)

    merged = _merge_all(parts)
    if save:
        mpath = output_dir / f"github_filtered_final_{base}.json"   # ★
        json.dump(merged, open(mpath,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print("✅ 최종 병합 결과 저장:", mpath)
    return merged
# ───────────── CLI ─────────────
if __name__ == "__main__":
    import sys
    mid = "bigscience/bloomz-560m"
    if len(sys.argv) > 1 and sys.argv[1]:
        mid = sys.argv[1]
    print("▶ 실행 모델:", mid)
    filter_github_features(mid)