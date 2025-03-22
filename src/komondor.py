import torch
from transformers import pipeline

def answer(user_prompt: str, system_prompt: str = "You are a pirate chatbot who always responds in pirate speak!", model_id: str = "meta-llama/Llama-3.2-1B-Instruct") -> str:
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=10,
    )
    return outputs[0]["generated_text"][-1]['content']
    
if __name__ == "__main__":
    print(answer(user_prompt="Who are you?"))