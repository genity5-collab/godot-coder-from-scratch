import argparse, os
from pathlib import Path
from huggingface_hub import HfApi, login

p = argparse.ArgumentParser()
p.add_argument('--model-dir', type=Path, default=Path('checkpoints/godot-coder-30m'))
p.add_argument('--repo-id', required=True, help='Example: username/godot-coder-30m')
p.add_argument('--private', action='store_true')
args = p.parse_args()
token = os.environ.get('HF_TOKEN')
if not token:
    raise SystemExit('Set HF_TOKEN in the notebook/runtime secret store; never commit it.')
login(token=token, add_to_git_credential=False)
api = HfApi(token=token)
api.create_repo(repo_id=args.repo_id, repo_type='model', private=args.private, exist_ok=True)
api.upload_folder(folder_path=str(args.model_dir), repo_id=args.repo_id, repo_type='model')
print(f'Uploaded {args.model_dir} to https://huggingface.co/{args.repo_id}')
