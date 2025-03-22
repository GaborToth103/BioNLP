"""Példa Python Script, mindent kipróbálunk."""

from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

model_name = "aaditya/OpenBioLLM-Llama3-8B"
file_path = 'data/output/example.txt'
prompt = "The future of medical AI is"

# .def fájlban definiált DATA elérése, kiírjuk, hogy mit látunk benne.
data_path = os.getenv("DATA") 
if data_path and os.path.isdir(data_path):
    files = os.listdir(data_path)
    for file in files:
        print(file)
else:
    print(f"Directory {data_path} does not exist or is not a valid path.")

# HF model elérése és szöveg generálás kipróbálása.
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generator(prompt, max_length=100, do_sample=True, truncation=True)
generated_text = output[0]["generated_text"]

# Kipróbáljuk, hogy az .sbatch fájlban definiált print és error csatornák elérhetőek-e, illetve képesek vagyunk-e írni egy fájlba. 
current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(file_path, 'w') as file:
    file.write(generated_text)
    file.write(f"\nThis should be placed into the {file_path} at {current_date}\n")
print(f"\nThis should be placed into the out.txt at {current_date}\n")
raise Exception(f"\nThis should be placed into the error.txt at {current_date}\n")