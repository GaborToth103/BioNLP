from model_loader import load_model, answer
from data_handler import *
from scoring import score_function


if __name__ == "__main__":
    scores = []
    data = DataHandler("data/dev/archehr-qa.xml", "data/dev/archehr-qa_key.json")
    system_prompt = f"You are a medical summarizer. Answer the patient question returning relevant sentence numbers in a list separated by spaces. Just write down relevant numbers and finish generation."        
    pipe = load_model()
    for case in data.cases.values():
        print(f"case-{case.case_id} processing...")
        model_answer: str = answer(pipe, case.summarize_for_llm(), system_prompt)
        print(model_answer)
        score: float = score_function(model_answer, case.sentences)                        
        scores.append(score)
        # print(case)
        print(f"score: {score}")
        # input("to confinue press ENTER...")
    print(scores)