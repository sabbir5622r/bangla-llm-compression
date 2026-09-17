# Bangla LLM Compression

Studying post-training quantization of small instruction-tuned LLMs on Bangla language-understanding tasks.

## Main comparison

- FP16
- INT8
- INT4


## Tasks

- Sentiment classification
- Natural language inference
- Fake-news classification

## Structure

- `data/` - dataset preparation
- `configs/` - experiment configuration
- `models/` - model loading
- `evaluation/` - evaluation
- `efficiency/` - efficiency measurement
- `experiments/` - experiment scripts
- `analysis/` - result analysis

## Setup

```bash
pip install -r requirements.txt