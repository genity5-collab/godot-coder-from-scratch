import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

p = argparse.ArgumentParser()
p.add_argument('--model', default='checkpoints/godot-coder-30m')
p.add_argument('--prompt', default='Write a Godot 4 script that extends Node2D and prints Hello Godot in _ready().')
p.add_argument('--max-new-tokens', type=int, default=160)
a = p.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model)
inputs = tok(a.prompt, return_tensors='pt')
out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=0.8, top_p=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
