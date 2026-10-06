# Datasets

This repository does not redistribute the raw datasets used in the study.
Users should obtain the datasets from their original sources and follow the
corresponding licenses and terms of use.

## Dataset Sources

### Stance Classification

The stance-classification dataset contains Bangla social-media comments
related to the July 2024 uprising in Bangladesh.

- Dataset: BJUS (Extended version of the July Revolution Sentiment Analysis Dataset Bangla)
- Paper: *Social Media Sentiments Analysis on the July Revolution in Bangladesh: A Hybrid Transformer Based Machine Learning Approach*
- Please contact the authors for the extended BJUS dataset.
### Natural Language Inference

We use the Bengali XNLI resource distributed by the CSE BUET NLP group.

- Dataset: `csebuetnlp/xnli_bn`
- License: CC BY-NC-SA 4.0
- Paper: *BanglaBERT: Language Model Pretraining and Benchmarks for
  Low-Resource Language Understanding Evaluation in Bangla*

### Fake News Detection

We use the BanFakeNews dataset introduced by Hossain et al. (2020).

- Dataset: BanFakeNews
- Paper: *BanFakeNews: A Dataset for Detecting Fake News in Bangla*
- Files used: `LabeledAuthentic-7K.csv` and `LabeledFake-1K.csv`


The raw dataset is not redistributed in this repository. Users should obtain
it from the original source and follow its applicable usage conditions.
## Preparation

After placing the raw files in `data/raw/`, run:

```bash
python data/prepare_datasets.py