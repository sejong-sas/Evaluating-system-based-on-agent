import requests
import re
import json
import fitz  # PyMuPDF
from typing import List, Dict
from pathlib import Path

def get_all_arxiv_ids(model_id: str) -> List[str]:
    """Extract all arXiv IDs from Hugging Face model tags"""
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

def download_arxiv_pdf(arxiv_id: str, save_path: str = None, output_dir: str | Path = ".") -> str:
    if not save_path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"arxiv_{arxiv_id}.pdf"
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    resp = requests.get(pdf_url)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(resp.content)
    print(f"📄 PDF saved: {save_path}")
    return str(save_path)

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def arxiv_fetcher_from_model(model_id: str, save_to_file: bool = True, output_dir: str | Path = ".") -> List[Dict[str, str]]:
    arxiv_ids = get_all_arxiv_ids(model_id)
    if not arxiv_ids:
        print(f"❌ No arXiv ID found for '{model_id}'.")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for arxiv_id in arxiv_ids:
        try:
            pdf_path = download_arxiv_pdf(arxiv_id, output_dir=output_dir)   # ★
            full_text = extract_text_from_pdf(pdf_path)
            results.append({"arxiv_id": arxiv_id, "full_text": full_text})
        except Exception as e:
            print(f"⚠️ Failed to process arXiv {arxiv_id}: {e}")

    if save_to_file:
        filename = output_dir / f"arxiv_fulltext_{model_id.replace('/', '_')}.json"   # ★
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"model_id": model_id, "full_texts": results}, f, indent=2, ensure_ascii=False)
        print(f"✅ Full paper text saved: {filename}")

    return results

# Standalone run example
if __name__ == "__main__":
    arxiv_fetcher_from_model("deepseek-ai/DeepSeek-R1")
