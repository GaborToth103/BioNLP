import torch
from transformers import pipeline

def load_model(model_id: str = "microsoft/Phi-4-mini-instruct") -> str:
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipe

def answer(pipe, user_prompt: str, system_prompt: str = "You are a pirate chatbot who always responds in pirate speak!") -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return pipe(
        messages,
        max_new_tokens=512,
    )[0]["generated_text"][-1]['content']
    
if __name__ == "__main__":
    print(answer(load_model(), user_prompt="who are you?")) 