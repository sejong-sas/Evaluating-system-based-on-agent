# web_search_test.py
import os, sys, json, traceback
from openai import OpenAI

# ✅ .env 로드
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

QUERY = "site:arxiv.org (technical report OR paper) large language model up to 3 urls"
OPENAI_MODEL_WEB = os.getenv("OPENAI_MODEL_WEB", "gpt-4.1-mini")          # Responses용
CHAT_SEARCH_MODEL = os.getenv("CHAT_SEARCH_MODEL", "gpt-4o-mini-search-preview")

def require_key(var="OPENAI_API_KEY"):
    if not os.getenv(var):
        raise SystemExit(f"{var} not found. Put it in .env or export it in your shell.")

def main():
    require_key("OPENAI_API_KEY")
    q = " ".join(sys.argv[1:]).strip() or QUERY
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # ... (나머지 테스트 코드 그대로)
