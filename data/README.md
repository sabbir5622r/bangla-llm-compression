# Dataset Preparation

This repository does not redistribute the raw datasets used in the study. Users should obtain the datasets from their original sources and place the required files under `data/raw/`.

The study evaluates three Bangla language-understanding tasks:

| Task | Dataset | Raw file(s) |
|---|---|---|
| Stance Classification | BJUS | `sentiment.csv` |
| Natural Language Inference | XNLI-BN | `bangla_nli.csv` |
| Fake News Detection | BanFakeNews | `LabeledAuthentic-7K.csv`, `LabeledFake-1K.csv` |

## Preparation

After placing the raw files in `data/raw/`, run:

```bash
python data/prepare_datasets.py