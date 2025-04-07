from model_loader import load_model, answer

class Character:
    """TODO This should be a BioNLP Character, with its own system prompt. It should call its own worker subprocess so it process inputs in batch structure."""
    def __init__(self, system_prompt: str, model_id: str):
        raise NotImplementedError()
        self.model_id = model_id
        self.system_prompt = system_prompt
    
    def call_worker():
        raise NotImplementedError()

if __name__ == "__main__":
    char = Character(f"You are a medical summarizer. Answer the patient question returning relevant sentence numbers in a list separated by spaces. Just write down relevant numbers and finish generation.", "microsoft/Phi-4-mini-instruct")
    char.call_worker()