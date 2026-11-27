import argparse
from pathlib import Path
import random

EXTENSIONS = {".gd", ".gdshader", ".tscn", ".tres", ".md", ".rst", ".txt"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    texts = []
    for path in sorted(args.input.rglob('*')):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            text = path.read_text(encoding='utf-8', errors='ignore').replace('\x00', '').strip()
            if len(text) >= 40:
                texts.append(f'### FILE: {path.name}\n{text}\n### END FILE')
    if not texts:
        raise SystemExit('No supported source files found. Add licensed files to data/raw/.')
    random.Random(args.seed).shuffle(texts)
    n = len(texts)
    train, valid = texts[:max(1, int(n*.9))], texts[max(1, int(n*.9)):max(2, int(n*.95))]
    test = texts[max(2, int(n*.95)):]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('train', train), ('validation', valid), ('test', test or valid)]:
        (args.output / f'{name}.txt').write_text('\n\n'.join(rows), encoding='utf-8')
    print(f'Wrote {len(train)} train, {len(valid)} validation, {len(test or valid)} test documents')

if __name__ == '__main__':
    main()
