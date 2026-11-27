"""Download official Godot documentation source for training.

The downloaded files are not committed to this repository. Attribution and the
upstream commit are recorded in data/raw/godot-docs-manifest.json.
"""
import argparse
import io
import json
import shutil
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('data/raw/godot-docs'))
    p.add_argument('--ref', default='master', help='Git branch or tag, default: master')
    args = p.parse_args()
    url = f'https://github.com/godotengine/godot-docs/archive/refs/heads/{args.ref}.tar.gz' if args.ref == 'master' else f'https://github.com/godotengine/godot-docs/archive/refs/tags/{args.ref}.tar.gz'
    print(f'Downloading official Godot docs: {url}')
    with urllib.request.urlopen(url) as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode='r:gz')
        root = archive.getnames()[0].split('/')[0]
        if args.output.exists():
            shutil.rmtree(args.output)
        args.output.mkdir(parents=True, exist_ok=True)
        count = 0
        for member in archive.getmembers():
            if not member.isfile() or not (member.name.endswith('.rst') or member.name.endswith('.md')):
                continue
            rel = Path(member.name).relative_to(root)
            if '..' in rel.parts:
                continue
            target = args.output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    manifest = {
        'source': 'https://github.com/godotengine/godot-docs',
        'ref': args.ref,
        'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
        'files': count,
        'license': 'CC BY 3.0 for content except classes/; MIT for classes/ derived content',
        'attribution': 'Juan Linietsky, Ariel Manzur and the Godot community',
        'notice': 'Review upstream LICENSE.txt and preserve attribution when redistributing derived datasets.'
    }
    manifest_path = args.output.parent / 'godot-docs-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Extracted {count} documentation files into {args.output}')
    print(f'Wrote provenance manifest to {manifest_path}')

if __name__ == '__main__':
    main()
