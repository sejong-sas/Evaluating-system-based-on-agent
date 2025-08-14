# inference.py
import os, sys, json, re, shlex, subprocess, time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────
# Environment
# ─────────────────────────────────────────
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
_client = OpenAI(api_key=_api_key)

OPENAI_MODEL = os.getenv("OPENAI_MODEL_INFER_PICK", "gpt-4o-mini")
ENC_RUN = "cp949" if os.name == "nt" else "utf-8"

# Default output dir (model folder). Upstream pipeline should set MODEL_OUTPUT_DIR.
_DEFAULT_OUTDIR = os.getenv("MODEL_OUTPUT_DIR") or os.getenv("CURRENT_MODEL_DIR") or "."

# ─────────────────────────────────────────
# GPT prompts
# ─────────────────────────────────────────
# 0) Does README contain a runnable local Python example?
_DETECT_EXEC_SYS = """
You are a precise classifier. Decide if the README contains a runnable *local Python* example (not servers/REST/vLLM serve).
Return JSON only:
{ "has_code": true|false, "reason": "<short reason>" }
"""

# 1) Extract one local Python example + pip installs — with hard rules the user requested
_PLAN_SYS = """
You are a helper that extracts one *runnable local Python* example from a README, plus the pip commands needed.

STRICT RULES (must follow all):
- Output a single JSON object only.
- Prefer a minimal transformers-based local inference script (no servers).
- Keep the original example code as-is EXCEPT for the following mandatory changes:
  1) Ensure the script is NON-INTERACTIVE:
     - Do NOT use input(), getpass(), or wait for user input.
     - If the original uses argparse/CLI flags, provide sane defaults inline so it runs by just `python run_tmp.py`.
  2) If there is any placeholder input (e.g., 'prompt = "여기에 인풋 프롬포트를 입력하세요"', 'YOUR_...', 'TODO', empty strings),
     replace it with a reasonable example input consistent with the README/model usage.
  3) END BY PRINTING the final textual result with `print(...)`. Make sure the printed content is the generated answer string (not the whole model object).
- No markdown code fences; put raw code in the "code" field.
- `pip_installs` must be a list of commands like "pip install ...". Include what's needed if imports appear.
- Only include ONE best example.

Schema:
{
  "pip_installs": ["pip install transformers>=4.46.0", ...],
  "filename": "run_tmp.py",
  "code_language": "python",
  "code": "<raw python code>",
  "notes": "optional"
}
"""

# 2) Minimal repair pass if the extracted plan still violates rules
_REPAIR_SYS = """
You are a minimal patcher. You receive:
- The README (context)
- A Python script extracted from that README

TASK: Return a minimally modified version of the script that satisfies ALL of these:
  A) Non-interactive: remove input()/getpass()/interactive prompts; if argparse/CLI is used, provide inline defaults so `python file.py` just runs.
  B) Replace any placeholder inputs (e.g., 'prompt = "여기에 인풋 프롬포트를 입력하세요"', 'YOUR_...', 'TODO', empty prompts) with reasonable example inputs consistent with the README/model.
  C) Ensure the final textual result is printed at the end with print(...).
  D) Keep everything else unchanged as much as possible (variable names, structure, imports).

Return JSON only:
{
  "code": "<patched code>",
  "notes": "what you changed in one sentence"
}
"""

def _ask_has_exec(readme: str) -> Dict[str, Any]:
    rsp = _client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _DETECT_EXEC_SYS},
            {"role": "user", "content": readme},
        ],
        temperature=0
    )
    try:
        return json.loads(rsp.choices[0].message.content)
    except Exception:
        return {"has_code": False, "reason": "Parsing error"}

def _ask_plan_from_readme(readme: str) -> Dict[str, Any]:
    rsp = _client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _PLAN_SYS},
            {"role": "user", "content": readme},
        ],
        temperature=0
    )
    try:
        return json.loads(rsp.choices[0].message.content)
    except Exception:
        return {}

def _ask_repair_code(readme: str, code: str, reasons: List[str]) -> Dict[str, Any]:
    user_msg = (
        "Reasons to repair:\n- " + "\n- ".join(reasons) +
        "\n\n--- ORIGINAL CODE START ---\n" + code + "\n--- ORIGINAL CODE END ---"
    )
    rsp = _client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _REPAIR_SYS},
            {"role": "user", "content": readme},
            {"role": "user", "content": user_msg}
        ],
        temperature=0
    )
    try:
        return json.loads(rsp.choices[0].message.content)
    except Exception:
        return {}

# ─────────────────────────────────────────
# pip normalization & execution
# ─────────────────────────────────────────
def _normalize_pip_cmd(cmd: str) -> List[str] | None:
    if not isinstance(cmd, str):
        return None
    cmd = cmd.strip().lstrip("!").replace("pip3", "pip")
    m = re.search(r"(?:python\s*-m\s+)?pip\s+install\s+(.+)", cmd, flags=re.I)
    if not m:
        return None
    args = shlex.split(m.group(1))
    return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"] + args

def _ensure_minimal_installs(plan: Dict[str, Any]) -> List[str]:
    installs = [c for c in (plan.get("pip_installs") or []) if isinstance(c, str)]
    code = plan.get("code") or ""
    if "transformers" in code and not any("transformers" in c for c in installs):
        installs.append("pip install transformers>=4.46.0")
    if re.search(r"\bimport\s+torch\b|\btorch\.", code) and not any(c.strip().startswith("pip install torch") for c in installs):
        installs.append("pip install torch")
    return installs

def _run_cmd(cmd_list: List[str], cwd: Path | None = None, timeout: int | None = None) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd_list,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding=ENC_RUN,
            timeout=timeout
        )
        return {
            "cmd": " ".join(cmd_list),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr
        }
    except subprocess.TimeoutExpired as e:
        return {
            "cmd": " ".join(cmd_list),
            "returncode": -9,
            "stdout": e.stdout or "",
            "stderr": f"TimeoutExpired: {e}"
        }
    except Exception as e:
        return {
            "cmd": " ".join(cmd_list),
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(e).__name__}: {e}"
        }

# ─────────────────────────────────────────
# Outdir helper
# ─────────────────────────────────────────
def _safe_outdir(output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return Path(_DEFAULT_OUTDIR)
    p = Path(output_dir)
    name = p.name
    if len(name) > 48 or re.search(r"[\\/:*?\"<>|]", name) or len(re.findall(r"\s", name)) > 6:
        return Path(_DEFAULT_OUTDIR)
    return p

# ─────────────────────────────────────────
# HF subfolder handling
# ─────────────────────────────────────────
def _load_first_hf_json(outdir: Path) -> Optional[Dict[str, Any]]:
    """Load the first huggingface_*.json from outdir, if any."""
    for fp in sorted(outdir.glob("huggingface_*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return None

def _detect_hf_subfolder(outdir: Path) -> Optional[str]:
    """
    Detect a subfolder for HF repos where config.json sits under a subdirectory.
    Returns the directory path (e.g., "subdir" or "nested/subdir") relative to repo root,
    or None if root has config.json or nothing to fix.
    """
    j = _load_first_hf_json(outdir)
    if not j:
        return None
    files = j.get("files") or []
    # If root has config.json → no need to inject
    if any(f == "config.json" for f in files):
        return None

    # Candidate dirs that contain a config.json
    cfg_dirs: Dict[str, int] = {}
    for f in files:
        if f.endswith("/config.json"):
            d = str(Path(f).parent).replace("\\", "/")
            cfg_dirs[d] = 0

    if not cfg_dirs:
        return None

    # Score candidates by likely weight files within each dir
    weight_suffixes = (
        ".safetensors", "pytorch_model.bin", "pytorch_model-00001-of",
        "consolidated.safetensors", "model.bin", ".gguf", ".ggml",
        "flax_model.msgpack", "tf_model.h5", "onnx/model.onnx"
    )
    for d in cfg_dirs.keys():
        d_prefix = d + "/"
        score = 0
        for f in files:
            if f.startswith(d_prefix) and f.endswith(weight_suffixes):
                score += 1
        cfg_dirs[d] = score

    # Pick the highest-scoring dir; if tie, choose lexicographically first
    best = None
    best_score = -1
    for d, s in sorted(cfg_dirs.items()):
        if s > best_score:
            best, best_score = d, s
    return best

def _inject_subfolder_arg(code: str, subfolder: str) -> tuple[str, bool]:
    """
    Inject subfolder="<subfolder>" into all .from_pretrained(...) calls
    if not already present. Returns (new_code, changed_flag).
    """
    if not subfolder or "from_pretrained" not in code:
        return code, False

    # If any call already has subfolder=..., leave that call unchanged.
    pat = re.compile(r"(\.from_pretrained\(\s*[^)]*?)\)", re.DOTALL)

    def repl(m):
        inner = m.group(1)
        if "subfolder=" in inner:
            return m.group(0)
        return inner + f', subfolder="{subfolder}")'

    new_code = pat.sub(repl, code)
    return (new_code, new_code != code)

# ─────────────────────────────────────────
# Simple code issue scan (for triggering repair)
# ─────────────────────────────────────────
def _scan_code_issues(code: str) -> List[str]:
    reasons: List[str] = []
    if not re.search(r"\bprint\s*\(", code):
        reasons.append("missing_print")
    if re.search(r"\binput\s*\(", code) or "getpass.getpass(" in code:
        reasons.append("interactive_input")
    if re.search(r'여기에\s*인풋|placeholder|YOUR_|enter your|paste your', code, flags=re.I):
        reasons.append("placeholder_input")
    # Very common pattern: empty prompt variable likely needing example text
    if re.search(r'\bprompt\s*=\s*["\']\s*["\']', code):
        reasons.append("empty_prompt")
    # argparse without defaults often causes interactivity or required args
    if "argparse.ArgumentParser(" in code and re.search(r"\.parse_args\(\)", code):
        reasons.append("argparse_requires_defaults")
    return reasons

# ─────────────────────────────────────────
# Main: new contract (two JSON files)
#   1) inference_plan_status.json
#   2) inference_output.json
# ─────────────────────────────────────────
def run_inference(readme: str, output_dir: str | Path | None = None, keep_code: bool = True) -> Path:
    """
    Returns the path to inference_output.json (2nd JSON).
    Also writes inference_plan_status.json (1st JSON) in the same folder.
    By default, the generated script (e.g., run_tmp.py) is preserved (keep_code=True).
    """
    outdir = _safe_outdir(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 0) detect if README contains a runnable local Python example
    detect = _ask_has_exec(readme)
    has_code = bool(detect.get("has_code"))
    detect_reason = detect.get("reason", "")

    plan_status = {
        "has_code": has_code,
        "detect_reason": detect_reason,
        "filename": None,
        "code_language": None,
        "code": "",
        "pip_installs": [],
        "install_results": [],           # [{cmd, success, returncode, stdout, stderr}]
        "execution_attempted": False,
        "run": {                         # preview only; full output is in inference_output.json
            "success": False,
            "returncode": None,
            "stdout_preview": "",
            "stderr_preview": ""
        },
        "notes": ""
    }

    output_json = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cmd": "",
        "returncode": None,
        "stdout": "",
        "stderr": ""
    }

    # If no code detected → save and exit early
    if not has_code:
        plan_path = outdir / "inference_plan_status.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_status, f, ensure_ascii=False, indent=2)

        out_path = outdir / "inference_output.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        print("⚠️ No runnable local example detected in README. Saved status/output skeletons.")
        return out_path

    # 1) ask GPT for an executable plan (already enforces print + non-interactive + placeholder replacement)
    plan = _ask_plan_from_readme(readme) or {}
    filename = plan.get("filename") or "run_tmp.py"
    code = (plan.get("code") or "").strip()
    code_language = plan.get("code_language") or "python"
    plan_status.update({
        "filename": filename,
        "code_language": code_language,
        "code": code
    })

    # If GPT failed to extract code, treat as no-code
    if not code:
        plan_status["has_code"] = False
        plan_status["detect_reason"] = "GPT could not extract a runnable local Python example."
        plan_path = outdir / "inference_plan_status.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_status, f, ensure_ascii=False, indent=2)

        out_path = outdir / "inference_output.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)
        print("⚠️ Plan extraction returned no code. Saved status/output skeletons.")
        return out_path

    # 1.5) Minimal repair pass if required
    issues = _scan_code_issues(code)
    if issues:
        patched = _ask_repair_code(readme, code, issues) or {}
        new_code = (patched.get("code") or "").strip()
        if new_code:
            code = new_code
            note = patched.get("notes") or ""
            plan_status["notes"] = (plan_status.get("notes") or "") + f' [repair] {note}'
            plan_status["code"] = code  # keep the final code snapshot in status

    # 2) normalize/augment installs
    plan_status["pip_installs"] = _ensure_minimal_installs(plan)

    # 3) HF subfolder auto-injection (if needed) — apply after repair
    subdir = _detect_hf_subfolder(outdir)
    if subdir:
        patched, changed = _inject_subfolder_arg(code, subdir)
        if changed:
            code = patched
            plan_status["notes"] = (plan_status.get("notes") or "") + \
                f' [auto] Injected subfolder="{subdir}" into from_pretrained(...) calls.'
            plan_status["code"] = code

    # 4) run pip installs (in outdir)
    install_logs: List[Dict[str, Any]] = []
    for raw in plan_status["pip_installs"]:
        norm = _normalize_pip_cmd(raw)
        if not norm:
            install_logs.append({
                "cmd": raw, "success": False, "returncode": -1, "stdout": "", "stderr": "Unrecognized pip command"
            })
            continue
        print(f"📦 Installing: {' '.join(norm)}")
        log = _run_cmd(norm, cwd=outdir, timeout=1200)
        log["success"] = (log.get("returncode") == 0)
        install_logs.append(log)
    plan_status["install_results"] = install_logs

    # 5) write code file and run it (code is preserved by default)
    codefile = outdir / filename
    with open(codefile, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"\n▶ Running script: {codefile.name} ...\n")
    plan_status["execution_attempted"] = True
    exec_log = _run_cmd([sys.executable, filename], cwd=outdir, timeout=1800)

    # populate output.json with full logs
    output_json.update({
        "cmd": exec_log.get("cmd", ""),
        "returncode": exec_log.get("returncode"),
        "stdout": exec_log.get("stdout", ""),
        "stderr": exec_log.get("stderr", "")
    })

    # in plan_status.json, only keep previews
    plan_status["run"]["returncode"] = exec_log.get("returncode")
    plan_status["run"]["success"] = (exec_log.get("returncode") == 0)
    plan_status["run"]["stdout_preview"] = (exec_log.get("stdout") or "")[:2000]
    plan_status["run"]["stderr_preview"] = (exec_log.get("stderr") or "")[:2000]

    # 6) optionally delete code if explicitly requested
    if not keep_code:
        try:
            codefile.unlink(missing_ok=True)
            print(f"🧹 Removed temporary script: {codefile.name}")
        except Exception:
            pass

    # 7) save JSONs
    plan_path = outdir / "inference_plan_status.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan_status, f, ensure_ascii=False, indent=2)

    out_path = outdir / "inference_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    # Console summary
    print("\n" + "-"*60)
    print(f"📄 Plan & status JSON: {plan_path}")
    print(f"📄 Output JSON: {out_path}")
    print(f"▶ Return code: {exec_log.get('returncode')}")
    print("-"*60)

    return out_path

# ─────────────────────────────────────────
# Standalone demo
# ─────────────────────────────────────────
if __name__ == "__main__":
    demo_readme = "Place README text here."
    # keep_code=True by default — the script file will be preserved in output_dir
    run_inference(demo_readme, output_dir=".", keep_code=True)
