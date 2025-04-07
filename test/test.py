from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "aaditya/OpenBioLLM-Llama3-70B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def ask_model(prompt) -> str:
    output = generator(prompt, max_length=4000, do_sample=True, truncation=True)
    generated_text = output[0]["generated_text"]
    print(generated_text)
    return ""

if __name__ == "__main__":
    prompt = (
        "Here is the following medical case:\n\n"

        "Patient Narrative:\n"
        "I spent yesterday in the ER with thumping heart beats i.e. palpitations. I had all blood work, "
        "a full panel, enzymes etc., EKG, chest x-ray, and TSH because I've had a total thyroidectomy. "
        "In October of last year, I had a walking stress test and all the same tests then, all of which "
        "are normal, showing no sign of any cardiac issues. My palpitations are benign, I'm told. Fine, "
        "how can I slow them or stop them without some antiarrhythmic meds? Of course, even though I've "
        "been told I'm fine, I feel them and sometimes worry which triggers the stress bug, then my "
        "chest gets tight, I have to take deep breaths to get out from under the stress. Is there "
        "anything I can do to relieve them? They really started when I started taking levothyroxine.\n\n"

        "Patient Question:\n"
        "My palpitations are benign, I'm told. Fine, how can I slow them or stop them without some "
        "antiarrhythmic meds?\n\n"

        "Clinician Question:\n"
        "What should he do to relieve palpitations and anxiety?\n\n"

        "Also, please observe the provided CSV file (bionlp_train) and its note and relevance columns. "
        "There you can see three types of classes: the notes can either be classified as essential, "
        "supplementary, and not relevant. For example, in case 3, there is a note that says "
        "'It is being strongly recommended that you follow up with Dr. [**First Name8 (NamePattern2)**] "
        "[**Last Name (NamePattern1)**] who is a doctor [**First Name (Titles)**] "
        "[**Last Name (Titles) 91506**] in traumatic brain injuries.' This is essential. "
        "The next note is 'His contact information has been provided to you.' This is not essential. "
        "The next note is 'Please report any:'. This is supplementary.\n\n"

        "After examining the CSV file like this, please classify the following notes to the clinical case above:\n"
        "1) This problem became irrelevant after the surgery, and endocrinology would like close follow-up in the outpatient setting to manage hypothyroidism s/p thyroidectomy.\n"
        "2) An ultrasound was performed which confirmed multi-nodular anatomy of goiter with tracheal compression; no biopsy was taken as there was already planning to remove the thyroid.\n"
        "3) After several family meetings, the patient and his mother (legal guardian) were in agreement that surgery would be indicated."
    )
    ask_model(prompt)
