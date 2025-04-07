from model_loader import load_model, answer_batch
import argparse
import pickle
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--system_prompt", type=str)
args = parser.parse_args()
pipe = load_model("microsoft/Phi-4-mini-instruct")
with open(args.data_path, "rb") as f:
    data = pickle.load(f)
answers = answer_batch(pipe, data, args.system_prompt)
with open(args.data_path, "wb") as f:
    pickle.dump(answers, f)
