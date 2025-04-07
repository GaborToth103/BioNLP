import torch
import gc
from transformers import pipeline
import time
from huggingface_hub import login
from dotenv import load_dotenv
import os
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

# List of models to switch between
MODEL_IDS = [
    "Writer/Palmyra-Med-70B",
    "aaditya/Llama3-OpenBioLLM-70B",
    "dmis-lab/llama-3-meerkat-70b-v1.0",
    "m42-health/Llama3-Med42-70B",
]

# 8b models
MODEL_IDS = [
    "aaditya/Llama3-OpenBioLLM-8B",
    "dmis-lab/llama-3-meerkat-8b-v1.0",
    "m42-health/Llama3-Med42-8B",
]


def load_model(model_id: str):
    """Loads the specified model and returns a text generation pipeline."""
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=HF_TOKEN
    )
    return pipe

def answer(pipe, user_prompt: str, system_prompt: str = "You are a pirate chatbot who always responds in pirate speak!") -> str:
    """Generates a response from the loaded model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = pipe(
        messages,
        max_new_tokens=16,
    )[0]["generated_text"][-1]['content']
    return response

def switch_models(user_prompt: str):
    """Cycles through models, generating responses from each."""
    responses = {}
    for model_id in MODEL_IDS:
        print(f"Loading model: {model_id}")
        pipe = load_model(model_id)
        response = answer(pipe, user_prompt)
        responses[model_id] = response
        print(f"{model_id}: {response}")
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(10)
    return responses

from collections import Counter

def voting_function(responses: list[list[int]]) -> list[int]:
    """Returns numbers that appear in at least two lists."""
    all_numbers = [num for response in responses for num in response]
    number_counts = Counter(all_numbers)
    
    return [num for num, count in number_counts.items() if count >= 2]

if __name__ == "__main__":
    user_input = "Who are you?"
    responses = switch_models(user_input)
    for model, response in responses.items():
        print(f"\nModel: {model}\nResponse: {response}\n")
