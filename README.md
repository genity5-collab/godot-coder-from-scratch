# Godot Coder From Scratch

A research project for training a small decoder-only language model from randomly initialized weights, specialized for Godot 4 and GDScript.

## Scope

This project trains a new tokenizer and a new GPT-style model. It does **not** load pretrained model weights. The first target is a 30M-parameter model that can learn GDScript syntax and Godot API patterns. It is an experiment, not a replacement for a large coding assistant.

## Model repository

The private Hugging Face model repository is [Dosmallelfso/godot-coder-30m](https://huggingface.co/Dosmallelfso/godot-coder-30m).

## Workflow

1. Add licensed Godot/GDScript sources to `data/raw/`.
2. Or download official Godot documentation with `python src/download_official_godot_docs.py`.
3. Run `python src/prepare_data.py` to clean and split the corpus.
4. Run `python src/train_tokenizer.py` to create the tokenizer.
5. Run `python src/train_model.py` on a GPU runtime in Colab or Kaggle.
6. Run `evaluation/generate.py` and Godot headless checks.
7. Upload the model to Hugging Face with `python src/upload_hf.py` after setting `HF_TOKEN`.

## Local setup

```bash
pip install -r requirements.txt
python src/prepare_data.py --input data/raw --output data/processed
python src/train_tokenizer.py --input data/processed/train.txt --output tokenizer/godot-bpe
```

## Colab/Kaggle

Open `notebooks/train_from_scratch.ipynb`, enable a GPU, and run all cells. Save checkpoints to persistent storage. Colab GPU type and runtime limits are dynamic; Kaggle sessions are also time-limited, so training resumes from checkpoints.

For the complete commands, smoke test, resume procedure, GPU out-of-memory fixes, and upload steps, read [`TRAINING.md`](TRAINING.md).

The official-data path is reproducible and does not commit the upstream docs into this repository:

```bash
python src/download_official_godot_docs.py --output data/raw/godot-docs --ref master
python src/prepare_data.py --input data/raw/godot-docs --output data/processed
```

Evaluation expects a real local checkpoint directory such as `checkpoints/godot-coder-30m`. If that directory does not exist, Transformers may interpret the path as a Hugging Face repository ID and return a confusing 401 error.

## Data licensing

Only use code and documentation that you created or have permission to redistribute. Record provenance in `data/SOURCES.md`. Do not commit secrets, private projects, model tokens, or large raw datasets.

## License

Code in this repository is MIT licensed. Training data retains its original licenses.
