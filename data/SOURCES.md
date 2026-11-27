# Data sources

Record one line per source before using it for training:

- source name:
- URL or project:
- license:
- version/commit:
- date collected:
- notes on preprocessing:

Do not add data unless its license permits the intended use and redistribution.

## Official Godot documentation

The training-data downloader uses the official source repository at https://github.com/godotengine/godot-docs. The repository states that content outside `classes/` is licensed CC BY 3.0 and attributed to Juan Linietsky, Ariel Manzur, and the Godot community. The `classes/` content is derived from the Godot source repository and is MIT licensed. The downloader records the selected ref and download time in `data/raw/godot-docs-manifest.json`.

Run this in Colab or Kaggle:

```bash
python src/download_official_godot_docs.py --output data/raw/godot-docs --ref master
python src/prepare_data.py --input data/raw/godot-docs --output data/processed
```
