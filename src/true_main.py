from data_handler import *
from scoring import *
from huggingface_hub import login
from dotenv import load_dotenv
import os
import re
from statistics import mean
from transformers import pipeline
import torch
import pandas as pd
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Run model with specified name.")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")

# Parse arguments
args = parser.parse_args()
model_name = args.model_name


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

def construct_prompt_refined_from_case(case: Case):
    sentences = [f"**{int(s.sentence_id)+1}:** {s.text}" for s in case.sentences]
    numbered_note = "\n".join(sentences)

    examples = """
Example:
Patient's Narrative: Took my 59 yo father to ER ultrasound discovered he had an aortic aneurysm. He had a salvage repair (tube graft). Long surgery / recovery for couple hours then removed packs. why did they do this surgery????? After this time he spent 1 month in hospital now sent home.

Patient's Question: why did they do this surgery?????

Clinician's Rephrased Question: Why did they perform the emergency salvage repair on him?

Clinical Note (numbered sentences):
**1:** He was transferred to the hospital on 2025-1-20 for emergent repair of his ruptured thoracoabdominal aortic aneurysm.
**2:** He was immediately taken to the operating room where he underwent an emergent salvage repair of ruptured thoracoabdominal aortic aneurysm with a 34-mm Dacron tube graft using deep hypothermic circulatory arrest.
**3:** Please see operative note for details which included cardiac arrest x2.
**4:** Postoperatively he was taken to the intensive care unit for monitoring with an open chest.
**5:** He remained intubated and sedated on pressors and inotropes.
**6:** On 2025-1-22, he returned to the operating room where he underwent exploration and chest closure.
**7:** On 1-25 he returned to the OR for abd closure JP/ drain placement/ feeding jejunostomy placed at that time for nutritional support.
**8:** Thoracoabdominal wound healing well with exception of very small open area mid wound that is @1cm around and 1/2cm deep, no surrounding erythema.
**9:** Packed with dry gauze and covered w/DSD.

Example Answer:
His aortic aneurysm was caused by the rupture of a thoracoabdominal aortic aneurysm, which required emergent surgical intervention (1). He underwent a complex salvage repair using a 34-mm Dacron tube graft and deep hypothermic circulatory arrest to address the rupture (2). The extended recovery time and hospital stay were necessary due to the severity of the rupture and the complexity of the surgery, though his wound is now healing well with only a small open area noted (8).
"""

    system_prompt = f"""You are a medical assistant tasked with generating a concise, helpful answer to a patient's health question using only information from the clinical note. Each statement in the answer must be grounded in specific sentences from the note.

{examples}

Now, the assistant should generate an answer for the following case:
"""

    user_prompt = f"""Patient's Narrative: {case.patient_narrative}

Patient's Question: {case.patient_question}

Clinician's Rephrased Question: {case.clinician_question}

Clinical Note (numbered sentences):
{numbered_note}

Instructions:
1. First, carefully identify which sentences are ESSENTIAL to answering the clinician's rephrased question. Focus on sentences that directly explain the medical reasoning, procedures performed, and clinical findings.
2. When writing your answer, ONLY include information from these essential sentences. Each statement in your answer MUST be supported by at least one citation.
3. For each statement in your answer, cite the specific sentence number(s) that support it using parentheses, e.g., "The procedure was successful (3, 5)."
4. Be very precise with your citations - only cite sentences that directly support each specific claim you make.

Your Answer:
"""

    return {
        'role': 'system', 
        'content': system_prompt
    }, {
        'role': 'user', 
        'content': user_prompt
    }

def generate_answer_prompt_from_case(case: Case):
    sentences = [f"**{int(s.sentence_id)+1}:** {s.text}" for s in case.sentences]
    numbered_note = "\n".join(sentences)

    examples = """<your example text here — same as before>"""

    system_prompt = f"""You are a medical assistant tasked with generating a helpful, concise answer to a patient's health question using only information from the clinical note.

{examples}

Now, the assistant should generate an answer for the following case:
"""

    user_prompt = f"""Patient's Narrative: {case.patient_narrative}

Patient's Question: {case.patient_question}

Clinician's Rephrased Question: {case.clinician_question}

Clinical Note (numbered sentences):
{numbered_note}

Instructions:
1. Answer the clinician's rephrased question directly and clearly.
2. Use only information found in the clinical note.

Your Answer:
"""

    return {
        'role': 'system', 
        'content': system_prompt
    }, {
        'role': 'user', 
        'content': user_prompt
    }

def extract_sentence_ids(answer: str) -> list[int]:
    matches = re.findall(r'\(([^)]+)\)', answer)
    ids = []
    for match in matches:
        ids.extend(map(int, re.findall(r'\d+', match)))
    return ids

def convert_indices_to_labels(predicted_ids: list[int], length: int, 
                              essential_label='essential', 
                              not_relevant_label='not-relevant') -> list[str]:
    y_pred = [not_relevant_label] * length
    for idx in predicted_ids:
        if 0 <= idx < length:
            y_pred[idx] = essential_label
    return y_pred

if __name__ == "__main__":
    ollama_use = False
    instruct = True
    
    if not ollama_use:
        pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        from ollama import chat
    data = DataHandler("data/dev/archehr-qa.xml", "data/dev/archehr-qa_key.json")

    f1s = []
    precisions = []
    recalls = []
    for case in data.cases.values():
        print("case started")
        prompt = construct_prompt_refined_from_case(case)
        messages = [prompt[0], prompt[1]]
        if not ollama_use:
            if instruct:
                answer = pipe(messages, max_new_tokens=512)[0]["generated_text"][-1]['content']
            else:
                answer = pipe(str(messages) + "\nYour answer: ", max_new_tokens=512)[0]["generated_text"]
        else:
            answer = chat(model='gemma3', messages=[prompt[0], prompt[1]])["message"]["content"]            
        predicted_sentence_ids = extract_sentence_ids(answer)
        y_true = [s.relevance.value for s in case.sentences]  # assuming Relevance enum values are like 'essential'
        y_pred = convert_indices_to_labels(predicted_sentence_ids, len(y_true))
        f1, precision, recall = f1_score(y_true, y_pred)
        print(f1, precision, recall)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    mean_f1s = mean(f1s)
    mean_precisions = mean(precisions)
    mean_recalls = mean(recalls)
    
    print(f"Average F1 score: {mean(f1s):.4f}")
    print(f"Average Precision: {mean(precisions):.4f}")
    print(f"Average Recall: {mean(recalls):.4f}")
    csv_data = {
        "model_name": [model_name],
        "mean_f1": [mean_f1s],
        "mean_precision": [mean_precisions],
        "mean_recall": [mean_recalls]
    }

    csv_file = "data/output/model_metrics.csv"

    # Check if file exists
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = pd.concat([df, pd.DataFrame(csv_data)], ignore_index=True)
    else:
        df = pd.DataFrame(csv_data)

    # Save to CSV
    df.to_csv(csv_file, index=False)