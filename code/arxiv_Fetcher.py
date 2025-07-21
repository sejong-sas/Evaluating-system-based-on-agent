import requests
import re
import json
import fitz  # PyMuPDF
from typing import List, Dict

def get_all_arxiv_ids(model_id: str) -> List[str]:
    """Hugging Face 모델 태그에서 arXiv ID 모두 추출"""
    url = f"https://huggingface.co/api/models/{model_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    tags = data.get("tags", [])
    arxiv_ids = []
    for tag in tags:
        if tag.startswith("arxiv:"):
            arxiv_ids.append(tag.replace("arxiv:", ""))
    return arxiv_ids

def download_arxiv_pdf(arxiv_id: str, save_path: str = None) -> str:
    if not save_path:
        save_path = f"arxiv_{arxiv_id}.pdf"
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    resp = requests.get(pdf_url)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(resp.content)
    print(f"📄 PDF 저장 완료: {save_path}")
    return save_path

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def arxiv_fetcher_from_model(model_id: str, save_to_file: bool = True) -> List[Dict[str, str]]:
    """모델에 등록된 모든 arXiv 논문의 본문 텍스트 추출"""
    arxiv_ids = get_all_arxiv_ids(model_id)
    if not arxiv_ids:
        print(f"❌ arXiv ID가 '{model_id}'에 없습니다.")
        return []

    results = []
    for arxiv_id in arxiv_ids:
        try:
            pdf_path = download_arxiv_pdf(arxiv_id)
            full_text = extract_text_from_pdf(pdf_path)
            results.append({
                "arxiv_id": arxiv_id,
                "full_text": full_text
            })
        except Exception as e:
            print(f"⚠️ arXiv {arxiv_id} 처리 실패: {e}")

    if save_to_file:
        filename = f"arxiv_fulltext_{model_id.replace('/', '_')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"model_id": model_id, "full_texts": results}, f, indent=2, ensure_ascii=False)
        print(f"✅ 논문 전체 본문 저장 완료: {filename}")

    return results


# 단독 실행 예시
if __name__ == "__main__":
    arxiv_fetcher_from_model("deepseek-ai/DeepSeek-R1")
