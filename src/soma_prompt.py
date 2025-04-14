def construct_prompt_refined(case_data):
    patient_narrative = case_data['patient_narrative']
    
    if isinstance(case_data['patient_question']['phrase'], list):
        patient_questions = [phrase['#text'] for phrase in case_data['patient_question']['phrase']]
        patient_question = " ".join(patient_questions)
    else:
        patient_question = case_data['patient_question']['phrase']['#text']
    
    clinician_question = case_data['clinician_question']
    
    sentences = []
    for sentence in case_data['note_excerpt_sentences']['sentence']:
        sentence_id = int(sentence['@id'])
        sentence_text = sentence['#text']
        sentences.append(f"**{sentence_id+1}:** {sentence_text}")
    
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
    
    prompt = f"""Task: Generate a concise, helpful answer to a patient's health question using only information from the clinical note. Each statement in your answer must be grounded in specific sentences from the note.

{examples}

Now, please generate an answer for the following case:

Patient's Narrative: {patient_narrative}

Patient's Question: {patient_question}

Clinician's Rephrased Question: {clinician_question}

Clinical Note (numbered sentences):
{numbered_note}

Instructions:
1. First, carefully identify which sentences are ESSENTIAL to answering the clinician's rephrased question. Focus on sentences that directly explain the medical reasoning, procedures performed, and clinical findings.

2. When writing your answer, ONLY include information from these essential sentences. Each statement in your answer MUST be supported by at least one citation.

3. For each statement in your answer, cite the specific sentence number(s) that support it using parentheses, e.g., "The procedure was successful (3, 5)."

4. Be very precise with your citations - only cite sentences that directly support each specific claim you make.

Your Answer:"""
    
    return prompt

def generate_answer_prompt(case_data):
    """
    Constructs a prompt for the first LLM to generate a patient-friendly answer
    based on the clinical notes.
    """
    patient_narrative = case_data['patient_narrative']
    
    if isinstance(case_data['patient_question']['phrase'], list):
        patient_questions = [phrase['#text'] for phrase in case_data['patient_question']['phrase']]
        patient_question = " ".join(patient_questions)
    else:
        patient_question = case_data['patient_question']['phrase']['#text']
    
    clinician_question = case_data['clinician_question']
    
    sentences = []
    for sentence in case_data['note_excerpt_sentences']['sentence']:
        sentence_id = int(sentence['@id'])
        sentence_text = sentence['#text']
        sentences.append(f"**{sentence_id+1}:** {sentence_text}")
    
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
His aortic aneurysm was caused by the rupture of a thoracoabdominal aortic aneurysm, which required emergent surgical intervention. He underwent a complex salvage repair using a 34-mm Dacron tube graft and deep hypothermic circulatory arrest to address the rupture. The extended recovery time and hospital stay were necessary due to the severity of the rupture and the complexity of the surgery, though his wound is now healing well with only a small open area noted.
"""
    
    prompt = f"""Task: Generate a helpful, concise answer to a patient's health question using only information from the clinical note.

{examples}

Now, please generate an answer for the following case:

Patient's Narrative: {patient_narrative}

Patient's Question: {patient_question}

Clinician's Rephrased Question: {clinician_question}

Clinical Note (numbered sentences):
{numbered_note}

Instructions:
1. Answer the clinician's rephrased question directly and clearly.
2. Use only information found in the clinical note.

Your Answer:"""
    
    return prompt

def generate_source_identification_prompt(case_data, generated_answer):
    """
    Constructs a prompt for the second LLM to identify which sentences from the 
    clinical note were used to support the generated answer.
    """
    sentences = []
    for sentence in case_data['note_excerpt_sentences']['sentence']:
        sentence_id = int(sentence['@id'])
        sentence_text = sentence['#text']
        sentences.append(f"**{sentence_id+1}:** {sentence_text}")
    
    numbered_note = "\n".join(sentences)
    
    prompt = f"""Task: Identify which sentences from the clinical note support statements in the patient answer.
    
-----
Example:

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

Example input text:
His aortic aneurysm was caused by the rupture of a thoracoabdominal aortic aneurysm, which required emergent surgical intervention. He underwent a complex salvage repair using a 34-mm Dacron tube graft and deep hypothermic circulatory arrest to address the rupture. The extended recovery time and hospital stay were necessary due to the severity of the rupture and the complexity of the surgery, though his wound is now healing well with only a small open area noted.

Essential Sentences: 1, 2, 8

-----

Now, please generate an answer for the following case:

Clinical Note (numbered sentences):
{numbered_note}

Input text:
{generated_answer}

Instructions:
1. Carefully analyze the answer and identify ALL sentences from the clinical note that directly support information in the answer.
2. Do not include sentences that contain information not referenced in the text.
3. List ONLY the sentence numbers (without any additional text) in a comma-separated format.
4. Your response should follow this format exactly:

Essential Sentences: [list of numbers]

For example: "Essential Sentences: 1, 3, 5, 7"
"""
    
    return prompt