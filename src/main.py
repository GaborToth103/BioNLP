from komondor import answer
from data_handler import DataHandler

if __name__ == "__main__":
    data = DataHandler("data/dev/archehr-qa.xml", "data/dev/archehr-qa_key.json")
    for case in data.cases.values():
            print(answer(user_prompt=case.summarize_for_llm()))
            input("\nyay\n")