#!/usr/bin/env python
"""
cleanup_folders.py
────────────────────────────────────────────────────────────
🔹 기능
  • make_model_dir() 가 만든 결과 폴더(예: bigscience_bloomz-560m)들을
    안전하게 한 번에 삭제합니다.

🔹 사용법
  1) 인자 없이 실행   → 현재 작업 디렉터리에서
                        '*/' 아래 JSON·PDF 파일을 포함한 폴더 자동 탐지 후 삭제
  2) 인자와 함께 실행 → python cleanup_folders.py folder1 folder2 ...
                        지정한 폴더만 삭제
────────────────────────────────────────────────────────────
"""

import sys, shutil, os, re
from pathlib import Path

# 결과 폴더 패턴: org_model (알파벳·숫자·_ 로만 구성)
RESULT_DIR_RE = re.compile(r"^[A-Za-z0-9_]+/[A-Za-z0-9_\-\.]+$")

def is_result_dir(p: Path) -> bool:
    """폴더 이름이 org_model 형태인지, 내부에 JSON/PDF 가 최소 1개 있는지 확인"""
    if not p.is_dir():
        return False
    # 폴더 이름이 'org_model' 패턴인지
    if not re.match(r"^[A-Za-z0-9_\-]+\_[A-Za-z0-9_\-\.]+$", p.name):
        return False
    # 내부에 JSON이나 PDF가 하나라도 있으면 결과 폴더로 간주
    for f in p.iterdir():
        if f.suffix.lower() in {".json", ".pdf"}:
            return True
    return False

def delete_dir(path: Path):
    try:
        shutil.rmtree(path)
        print(f"🗑️  삭제 완료: {path}")
    except Exception as e:
        print(f"⚠️  {path} 삭제 실패:", e)

def main():
    if len(sys.argv) > 1:
        targets = [Path(name) for name in sys.argv[1:]]
    else:
        # 👉 인자 없으면 사용자에게 한 줄 입력받기
        name = input("삭제할 폴더 이름(여러 개는 공백으로 구분): ").strip()
        if not name:
            print("취소되었습니다."); sys.exit(0)
        targets = [Path(n) for n in name.split()]


    if not targets:
        print("삭제할 대상 폴더가 없습니다.")
        return

    print("❗ 삭제 대상 폴더:")
    for t in targets:
        print("  -", t.resolve())
    confirm = input("\n정말 삭제할까요? (y/N) ").strip().lower()
    if confirm != "y":
        print("취소되었습니다.")
        return

    for t in targets:
        delete_dir(t)

if __name__ == "__main__":
    main()
