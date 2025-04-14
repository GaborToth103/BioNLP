from model_loader import load_model, answer

class Character:
    """TODO This should be a BioNLP Character, with its own system prompt. It should call its own worker subprocess so it process inputs in batch structure."""
    def __init__(self, name, system_prompt: str, pipe):
        self.name = name
        self.system_prompt = system_prompt
        self.pipe = pipe

    def answer(self, user_prompt: str, history: str) -> str:
        """Generates a response from the loaded model."""
        response = answer(self.pipe, user_prompt, history, self.system_prompt)
        return response
    
    def call_worker():
        raise NotImplementedError()

def consilium():
    pipe = load_model("microsoft/Phi-4-mini-instruct")
    chars: list[Character] = []
    chars.append(Character("medical summarizer", f"You are a medical summarizer. discuss the patient question returning relevant sentence numbers in a list separated by spaces. Just write down relevant numbers and finish generation.", pipe))
    chars.append(Character("sceptical doctor", f"You are a sceptical doctor, discussing the relevant points with other medical staff. Tell them your thoughts.", pipe))
    chars.append(Character("helpful assistant", f"You are a helpful assistant. Discuss the patient question with the medical summarizer and sceptical doctor. Tell them your thoughts.", pipe))
    user_prompt = "What is the treatment for a common cold?"
    history = []
    history.append({"role": "user", "content": user_prompt})
    for turn in range(2):
        for char in chars:
            response = char.answer("Tell us something!", history)
            history.append({"role": char.name, "content": response})
    for history_item in history:
        print(f"{history_item['role']}: {history_item['content']}")
    
if __name__ == "__main__":
    consilium()