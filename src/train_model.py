import argparse
import inspect
from pathlib import Path

from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)


def latest_checkpoint(output: Path):
    checkpoints = sorted(output.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    return checkpoints[-1] if checkpoints else None


def main():
    p = argparse.ArgumentParser(description='Train a new GPT-style Godot model from random weights.')
    p.add_argument('--data', type=Path, default=Path('data/processed'))
    p.add_argument('--tokenizer', type=Path, default=Path('tokenizer/godot-bpe/tokenizer.json'))
    p.add_argument('--output', type=Path, default=Path('checkpoints/godot-coder-30m'))
    p.add_argument('--context-length', type=int, default=1024)
    p.add_argument('--epochs', type=float, default=3)
    p.add_argument('--max-steps', type=int, default=-1, help='Use a positive value for a short smoke test.')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--grad-accumulation', type=int, default=16)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--save-steps', type=int, default=500)
    p.add_argument('--eval-steps', type=int, default=500)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if not args.tokenizer.exists():
        raise SystemExit(f'Tokenizer not found: {args.tokenizer}. Run train_tokenizer.py first.')
    for split in ('train.txt', 'validation.txt'):
        if not (args.data / split).exists():
            raise SystemExit(f'Dataset split not found: {args.data / split}. Run prepare_data.py first.')

    set_seed(args.seed)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.tokenizer),
        bos_token='<bos>', eos_token='<eos>', unk_token='<unk>', pad_token='<pad>'
    )
    raw = load_dataset('text', data_files={
        'train': str(args.data / 'train.txt'),
        'validation': str(args.data / 'validation.txt'),
    })

    def tokenize(batch):
        return tokenizer(batch['text'], add_special_tokens=False)

    tokenized = raw.map(tokenize, batched=True, remove_columns=['text'], desc='Tokenizing')

    def pack(batch):
        joined = {key: sum(batch[key], []) for key in batch}
        total = (len(joined['input_ids']) // args.context_length) * args.context_length
        return {
            key: [values[i:i + args.context_length] for i in range(0, total, args.context_length)]
            for key, values in joined.items()
        }

    packed = tokenized.map(pack, batched=True, desc='Packing token blocks')
    if len(packed['train']) == 0:
        raise SystemExit('No complete token blocks were produced. Add more training data or lower --context-length.')
    if len(packed['validation']) == 0:
        raise SystemExit('No validation token blocks were produced. Add more validation data or lower --context-length.')

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.context_length,
        n_ctx=args.context_length,
        n_embd=512,
        n_layer=8,
        n_head=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)  # Deliberately random initialization.
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    use_fp16 = __import__('torch').cuda.is_available()
    supported_args = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs = dict(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=20,
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=use_fp16,
        report_to='none',
        dataloader_num_workers=2,
    )
    if 'warmup_ratio' in supported_args:
        training_kwargs['warmup_ratio'] = 0.02
    elif 'warmup_steps' in supported_args:
        training_kwargs['warmup_steps'] = 100
    if 'eval_strategy' in supported_args:
        training_kwargs['eval_strategy'] = 'steps'
    elif 'evaluation_strategy' in supported_args:
        training_kwargs['evaluation_strategy'] = 'steps'
    training = TrainingArguments(**{
        key: value for key, value in training_kwargs.items() if key in supported_args
    })
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=packed['train'],
        eval_dataset=packed['validation'],
        data_collator=collator,
    )
    checkpoint = latest_checkpoint(args.output) if args.resume else None
    print(f'Train blocks: {len(packed["train"])}; validation blocks: {len(packed["validation"])}')
    print(f'Model parameters: {model.num_parameters():,}; random initialization: yes')
    print(f'Resuming from: {checkpoint or "scratch"}')
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f'Saved model to {args.output.resolve()}')


if __name__ == '__main__':
    main()
