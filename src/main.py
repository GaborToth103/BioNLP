from data_handler import *
import subprocess
import pickle
import re
import os
from dotenv import load_dotenv

def extract_numbers(input_string: str) -> list[int]:
    # Regex pattern to match any integer
    numbers = re.findall(r'\d+', input_string)
    
    # Convert matched strings to integers
    return [int(num) for num in numbers]

def call_worker(file_path, system_prompt, data) -> list[str]:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    subprocess.run(["python", "src/model_subprocess.py", "--data_path", file_path, "--system_prompt", system_prompt], capture_output=True, text=True)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def concilium_loop(loop_count: int = 2) -> list:
    raise NotImplementedError()
    return []

if __name__ == "__main__":
    load_dotenv()
    data = DataHandler(f"{os.getenv("DATA")}/dev/archehr-qa.xml", f"{os.getenv("DATA")}/dev/archehr-qa_key.json")
    system_prompt = f"You are a medical summarizer. Answer the patient question returning relevant sentence numbers in a list separated by spaces. Just write down relevant numbers and finish generation."        
    file_path = "data/data.pkl"
    summaries = [case.summarize_for_llm() for case in data.cases.values()]
    answers = call_worker(file_path, system_prompt, summaries)
    for answer in answers:
        numbers = extract_numbers(answer.split("\n")[-1])
        print(numbers)
