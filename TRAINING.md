# Full training workflow

This is a from-scratch model: the GPT model and tokenizer are created locally from new weights. The Hugging Face model repository is only a destination for checkpoints; it is not the source of the model.

## Colab setup

Create a new notebook, select a GPU, and run these cells in order.

```python
!nvidia-smi
!git clone https://github.com/genity5-collab/godot-coder-from-scratch.git
%cd godot-coder-from-scratch
!pip install -q -U torch transformers datasets tokenizers accelerate huggingface_hub safetensors
```

Download official Godot documentation and build the dataset:

```python
!python src/download_official_godot_docs.py --output data/raw/godot-docs --ref master
!python src/prepare_data.py --input data/raw/godot-docs --output data/processed
!wc -c data/processed/*.txt
```

Train the tokenizer from scratch:

```python
!python src/train_tokenizer.py \
  --input data/processed/train.txt \
  --output tokenizer/godot-bpe \
  --vocab-size 12000
```

Run a small smoke test first. This checks the whole pipeline and should finish much sooner than the real run:

```python
!python src/train_model.py \
  --data data/processed \
  --tokenizer tokenizer/godot-bpe/tokenizer.json \
  --output checkpoints/smoke-test \
  --context-length 512 \
  --epochs 1 \
  --max-steps 50 \
  --batch-size 1 \
  --grad-accumulation 8 \
  --save-steps 25 \
  --eval-steps 25
```

Test the smoke-test checkpoint. The local directory must exist before running this command:

```python
!python evaluation/generate.py \
  --model checkpoints/smoke-test \
  --prompt "Write a Godot 4 Node2D script that prints Hello Godot."
```

Start the real training run:

```python
!python src/train_model.py \
  --data data/processed \
  --tokenizer tokenizer/godot-bpe/tokenizer.json \
  --output checkpoints/godot-coder-30m \
  --context-length 1024 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accumulation 16 \
  --learning-rate 0.0003 \
  --save-steps 250 \
  --eval-steps 250
```

If Colab or Kaggle disconnects, reconnect, clone the repository again, restore the `checkpoints/godot-coder-30m` directory from Drive/Kaggle output, and resume:

```python
!python src/train_model.py \
  --data data/processed \
  --tokenizer tokenizer/godot-bpe/tokenizer.json \
  --output checkpoints/godot-coder-30m \
  --resume
```

A checkpoint must actually exist inside the output directory for `--resume` to work. The script automatically selects the latest `checkpoint-*` directory.

## Kaggle setup

Create a Kaggle Notebook, set **Settings → Accelerator → GPU**, and run:

```python
!nvidia-smi
!git clone https://github.com/genity5-collab/godot-coder-from-scratch.git
%cd godot-coder-from-scratch
!pip install -q -U torch transformers datasets tokenizers accelerate huggingface_hub safetensors
```

Then run the same official-data, tokenizer, smoke-test, and real-training commands above. To preserve results, use **Save Version → Save & Run All**, and download the `checkpoints/` directory from the notebook output before the session ends.

## Upload to Hugging Face

Create a new Kaggle or Colab secret named `HF_TOKEN`. Use a replacement write token, not the token previously pasted into chat.

Colab:

```python
from google.colab import userdata
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
```

Kaggle:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
```

Upload only after the local directory exists:

```python
!test -f checkpoints/godot-coder-30m/config.json
!python src/upload_hf.py \
  --model-dir checkpoints/godot-coder-30m \
  --repo-id Dosmallelfso/godot-coder-30m \
  --private
```

## Common errors

If Transformers requests `https://huggingface.co/checkpoints/godot-coder-30m`, the local checkpoint directory is missing. Run the training command first, check it with `!ls -la checkpoints/godot-coder-30m`, and then run evaluation with the same local path.

If the tokenizer command says that no files were found, check that `data/processed/train.txt` exists and contains text. If training says that no complete token blocks were produced, lower `--context-length 512` or add more documentation data.

If the GPU runs out of memory, reduce `--batch-size` to `1`, increase `--grad-accumulation` to `32`, and use `--context-length 512`. The effective batch size is approximately `batch-size × grad-accumulation`.
