# inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def run_inference(model_id: str, prompt: str = "Explain LLMs simply.") -> str:
    print(f"🤖 모델 로드 중: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n🧠 생성 결과:")
    print(result)
    return result

# CLI 테스트용
if __name__ == "__main__":
    run_inference("deepseek-ai/deepseek-v3", "What is deep learning?")
