import xml.etree.ElementTree as ET
from enum import Enum
import json
import re

class Relevance(Enum):
    ESSENTIAL = "essential"
    SUPPLEMENTARY = "supplementary"
    NOT_RELEVANT = "not-relevant"

class Sentence:
    def __init__(self, sentence_id, text, relevance=Relevance.NOT_RELEVANT):
        self.sentence_id = sentence_id
        self.text = text
        self.relevance: Relevance = relevance

    def __str__(self):
        return f"Sentence {self.sentence_id} ({self.relevance.value}): {self.text}"

class Case:
    def __init__(self, case_id, patient_narrative, patient_question, sentences):
        self.case_id = case_id
        self.patient_narrative = patient_narrative
        self.patient_question = patient_question
        self.sentences = sentences

    def __str__(self) -> str:
        return f"Case {self.case_id}:\n" \
               f"Patient Narrative: {self.patient_narrative}\n" \
               f"Patient Question: {self.patient_question}\n" \
               f"{'\n'.join(str(s) for s in self.sentences)}\n"
    
    def summarize_for_llm(self) -> str:        
        return f"Patient Narrative: {self.patient_narrative}\n" \
               f"Patient Question: {self.patient_question}\n" \
               f"{' '.join(str(s.text) for s in self.sentences)}\n" \
               f"The list of relevant sentences:"


class DataHandler:
    def __init__(self, xml_path: str, json_path: str):
        self.cases = self.load_cases_from_xml(xml_path)
        self.update_cases_with_relevance(self.cases, json_path)
                
    def load_cases_from_xml(self, xml_path: str) -> dict[str, Case]:
        # Load and parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        """Parses the XML and returns a dictionary of Case objects indexed by case_id."""
        cases: dict[str, Case] = {}
        
        for case in root.findall("case"):
            case_id = case.get("id")
            patient_narrative = case.find("patient_narrative").text.strip()
            patient_question = case.find("patient_question/phrase").text.strip()

            sentences = []
            for sentence in case.findall("note_excerpt_sentences/sentence"):
                sentence_id = sentence.get("id")
                text = sentence.text.strip()
                sentences.append(Sentence(sentence_id, self.clean_text(text)))

            cases[case_id] = Case(case_id, patient_narrative, patient_question, sentences)

        return cases

    def update_cases_with_relevance(self, cases: dict[str, Case], json_path: str):
        """Updates sentence relevance from the JSON file."""
        with open(json_path, "r") as f:
            relevance_data = json.load(f)

        for entry in relevance_data:
            case_id = entry["case_id"]
            if case_id in cases:  # Ensure the case exists in our data
                for ans in entry["answers"]:
                    sentence_id = ans["sentence_id"]
                    relevance = Relevance(ans["relevance"])  # Convert string to enum

                    # Find the matching sentence by ID
                    for sentence in cases[case_id].sentences:
                        if sentence.sentence_id == sentence_id:
                            sentence.relevance = relevance
                            break  # No need to continue searching

    def clean_text(self, text: str) -> str:
        """Removes extra spaces, newlines, and special whitespace characters."""
        return re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces/newlines with a single space

if __name__ == "__main__":
    data = DataHandler("data/dev/archehr-qa.xml", "data/dev/archehr-qa_key.json")
    for case in data.cases.values():
        print(case)
        # print(case.summarize_for_llm())
        input()