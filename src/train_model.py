import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainingArguments, Trainer, DataCollatorForLanguageModeling


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/processed'))
    p.add_argument('--tokenizer', type=Path, default=Path('tokenizer/godot-bpe/tokenizer.json'))
    p.add_argument('--output', type=Path, default=Path('checkpoints/godot-coder-30m'))
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    tok = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer), bos_token='<bos>', eos_token='<eos>', unk_token='<unk>', pad_token='<pad>')
    raw = load_dataset('text', data_files={'train': str(args.data/'train.txt'), 'validation': str(args.data/'validation.txt')})
    def enc(batch):
        return tok(batch['text'], truncation=True, max_length=1024)
    ds = raw.map(enc, batched=True, remove_columns=['text'])
    config = GPT2Config(vocab_size=tok.vocab_size, n_positions=1024, n_ctx=1024, n_embd=512, n_layer=8, n_head=8, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    model = GPT2LMHeadModel(config)  # Random initialization: no from_pretrained call.
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    training = TrainingArguments(output_dir=str(args.output), per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=16, learning_rate=3e-4, weight_decay=0.1, warmup_ratio=0.02, num_train_epochs=3, logging_steps=20, eval_strategy='steps', eval_steps=500, save_steps=500, save_total_limit=3, fp16=True, report_to='none')
    trainer = Trainer(model=model, args=training, train_dataset=ds['train'], eval_dataset=ds['validation'], data_collator=collator)
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(str(args.output)); tok.save_pretrained(str(args.output))

if __name__ == '__main__':
    main()
