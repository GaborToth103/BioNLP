from ollama import chat
from data_handler import *
from scoring import *

def construct_prompt_refined_from_case(case: Case):
    sentences = [f"**{int(s.sentence_id)+1}:** {s.text}" for s in case.sentences]
    numbered_note = "\n".join(sentences)

    examples = """<your example text here — same as before>"""

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

import re

def extract_sentence_ids(answer: str) -> list[int]:
    return list(map(int, re.findall(r'\((\d+)\)', answer)))


def convert_indices_to_labels(predicted_ids: list[int], length: int, 
                              essential_label='essential', 
                              not_relevant_label='not-relevant') -> list[str]:
    y_pred = [not_relevant_label] * length
    for idx in predicted_ids:
        if 0 <= idx < length:
            y_pred[idx] = essential_label
    return y_pred

from statistics import mean

if __name__ == "__main__":
    data = DataHandler("data/dev/archehr-qa.xml", "data/dev/archehr-qa_key.json")
    f1s = []
    precisions = []
    recalls = []
    for case in data.cases.values():
        prompt = construct_prompt_refined_from_case(case)
        answer = chat(model='gemma3', messages=[prompt[0], prompt[1]])["message"]["content"]
        predicted_sentence_ids = extract_sentence_ids(answer)
        y_true = [s.relevance.value for s in case.sentences]  # assuming Relevance enum values are like 'essential'
        y_pred = convert_indices_to_labels(predicted_sentence_ids, len(y_true))
        f1, precision, recall = f1_score(y_true, y_pred)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    print(f"Average F1 score: {mean(f1s):.4f}")
    print(f"Average Precision: {mean(precisions):.4f}")
    print(f"Average Recall: {mean(recalls):.4f}")