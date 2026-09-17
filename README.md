# Bangla LLM Compression

Research code for studying post-training quantization of small instruction-tuned LLMs on Bangla language-understanding tasks.

## Main comparison

- FP16
- INT8
- INT4

## Tasks

1. Bangla sentiment classification
2. Bangla fake-news classification
3. Bangla natural-language inference

## Development workflow

VS Code -> Git -> GitHub -> Kaggle -> GPU experiments -> results -> GitHub

## First-stage setup

Place the three local CSV files under:

`data/raw/`

Expected files:

- `sentiment.csv`
- `LabeledAuthentic-7K.csv`
- `LabeledFake-1K.csv`

Then run:

```powershell
python -m pip install -r requirements.txt
python data/inspect_dataset.py
python data/prepare_datasets.py
```

The NLI dataset is downloaded from Hugging Face by the preparation script.

## Important

Do not commit raw datasets. The `.gitignore` excludes `data/raw/`.
