import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--vocab-size', type=int, default=12000)
    args = p.parse_args()
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size, min_frequency=2,
        special_tokens=['<pad>', '<unk>', '<bos>', '<eos>', '<|user|>', '<|assistant|>'])
    tokenizer.train([str(args.input)], trainer)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output / 'tokenizer.json'))
    print(f'Saved tokenizer to {args.output}')

if __name__ == '__main__':
    main()
