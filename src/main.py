from data_handler import *
import subprocess
import pickle
import re
import os
from dotenv import load_dotenv

MODEL_IDS = [
    "Writer/Palmyra-Med-70B",
    "aaditya/Llama3-OpenBioLLM-70B",
    "dmis-lab/llama-3-meerkat-70b-v1.0",
    "m42-health/Llama3-Med42-70B",
]

def extract_numbers(input_string: str) -> list[int]:
    numbers = re.findall(r'\d+', input_string)
    return [int(num) for num in numbers]

def call_worker(file_path, system_prompt, data, model_id) -> list[str]:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    subprocess.run(["python", "src/model_subprocess.py", "--data_path", file_path, "--system_prompt", system_prompt], capture_output=True, text=True)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def concilium_loop(summaries: list[str]) -> list:
    """TODO Loop through the models and return the most common numbers."""
    raise NotImplementedError()
    summaries = []
    for model_id in MODEL_IDS:
        summaries = call_worker(file_path, system_prompt, summaries, model_id)
    return summaries

if __name__ == "__main__":
    load_dotenv()
    data = DataHandler(f"{os.getenv("DATA")}/dev/archehr-qa.xml", f"{os.getenv("DATA")}/dev/archehr-qa_key.json")
    system_prompt = f"You are a medical summarizer. Answer the patient question returning relevant sentence numbers in a list separated by spaces. Just write down relevant numbers and finish generation."        
    file_path = "data/data.pkl"
    summaries = [case.summarize_for_llm() for case in data.cases.values()]
    answers = call_worker(file_path, system_prompt, summaries, "microsoft/Phi-4-mini-instruct")
    for answer in answers:
        numbers = extract_numbers(answer.split("\n")[-1])
        print(numbers)
    # final_answer = concilium_loop(summaries)