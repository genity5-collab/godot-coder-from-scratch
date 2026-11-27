import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--model', default='checkpoints/godot-coder-30m', help='Local output directory or Hugging Face repo_id.')
p.add_argument('--prompt', default='Write a Godot 4 script that extends Node2D and prints Hello Godot in _ready().')
p.add_argument('--max-new-tokens', type=int, default=160)
p.add_argument('--temperature', type=float, default=0.8)
a = p.parse_args()

model_path = Path(a.model)
looks_like_local_path = a.model.startswith(('checkpoints/', './', '../', '/'))
if not model_path.exists() and looks_like_local_path:
    raise SystemExit(
        f'Local model directory not found: {a.model}\n'
        'Train first with src/train_model.py, or pass a valid Hugging Face repo_id.'
    )
source = str(model_path.resolve()) if model_path.exists() else a.model
tok = AutoTokenizer.from_pretrained(source)
model = AutoModelForCausalLM.from_pretrained(source)
inputs = tok(a.prompt, return_tensors='pt')
out = model.generate(
    **inputs,
    max_new_tokens=a.max_new_tokens,
    do_sample=True,
    temperature=a.temperature,
    top_p=0.95,
    pad_token_id=tok.eos_token_id,
)
print(tok.decode(out[0], skip_special_tokens=True))
