# 📚 Datasets

This repository does not redistribute the raw datasets used in the study.

<p align="justify">
Users should obtain the datasets from their original sources and follow the corresponding licenses and terms of use. After obtaining the datasets, the provided preprocessing and splitting scripts can be used to reconstruct the data used by the evaluation pipeline.
</p>

---

## Dataset Sources

### Stance Classification

<p align="justify">
The stance-classification dataset contains Bangla social-media comments related to the July 2024 uprising in Bangladesh.
</p>

- **Dataset:** BJUS (Extended version of the July Revolution Sentiment Analysis Dataset Bangla)
- **Paper:** *Social Media Sentiments Analysis on the July Revolution in Bangladesh: A Hybrid Transformer Based Machine Learning Approach*

### Natural Language Inference

We use the Bengali XNLI resource distributed by the CSE BUET NLP group.

- **Dataset:** `csebuetnlp/xnli_bn`
- **License:** CC BY-NC-SA 4.0
- **Paper:** *BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla*

### Fake News Detection

We use the BanFakeNews dataset introduced by Hossain et al. (2020).

- **Dataset:** BanFakeNews
- **Paper:** *BanFakeNews: A Dataset for Detecting Fake News in Bangla*
- **Files used:** `LabeledAuthentic-7K.csv` and `LabeledFake-1K.csv`

<p align="justify">
The raw datasets are not redistributed in this repository. Users should obtain them from their original sources and follow their applicable usage conditions.
</p>

---

## Expected Data Structure

After obtaining the datasets, place the required raw files under `data/raw/`.

```text
data/
│
├── raw/
│   ├── sentiment.csv
│   ├── bangla_nli.csv
│   ├── LabeledAuthentic-7K.csv
│   └── LabeledFake-1K.csv
│
├── processed/
│   ├── stance.csv
│   ├── nli.csv
│   ├── fake_news.csv
│   │
│   └── splits/
│       ├── stance/
│       │   ├── train.csv
│       │   ├── dev.csv
│       │   └── test.csv
│       │
│       ├── nli/
│       │   ├── train.csv
│       │   ├── dev.csv
│       │   └── test.csv
│       │
│       └── fake_news/
│           ├── train.csv
│           ├── dev.csv
│           └── test.csv
│
├── README.md
├── prepare_datasets.py
└── create_splits.py
```

> The files under `data/processed/` are generated automatically and should not be added manually.

---

## Preparation

### 1. Prepare the Datasets

After placing the required raw files in `data/raw/`, run:

```bash
python data/prepare_datasets.py
```

<p align="justify">
This script preprocesses the three datasets and writes the prepared task-level files under `data/processed/`.
</p>

### 2. Create the Experimental Splits

Run:

```bash
python data/create_splits.py
```

The processed splits are written to:

```text
data/processed/splits/
```

The splitting procedure uses a fixed random seed of `42`.

### Generated Split Sizes

| Task | Train | Development | Test | Total |
|---|---:|---:|---:|---:|
| Stance | 7,850 | 981 | 982 | 9,813 |
| NLI | 10,024 | 1,251 | 1,253 | 12,528 |
| Fake News | 6,800 | 850 | 851 | 8,501 |

<p align="justify">
Stance and fake-news data use stratified splitting, while NLI uses group-aware splitting to prevent examples derived from the same premise from appearing across different partitions.
</p>

---

## Data Version Control

<p align="justify">
Raw and processed dataset files are intentionally excluded from Git version control because the original datasets are governed by their respective sources and usage conditions.
</p>

Only `.gitkeep` files are tracked inside `data/raw/` and `data/processed/` so that the expected directory structure remains visible after cloning the repository.

The preprocessing and splitting scripts are provided to reconstruct the processed data locally.