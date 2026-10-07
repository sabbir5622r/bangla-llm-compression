<div align="center">

# 🗜️ Bangla LLM Compression

### How Much Can We Compress Small LLMs? Quantization Trade-offs for Low-Resource Bangla Language Understanding

**Evaluating FP16, INT8, and 4-bit NF4 quantization across Qwen2.5 and Falcon3 models on Bangla language understanding.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E)](https://huggingface.co/docs/transformers/)
[![Qwen2.5](https://img.shields.io/badge/Model-Qwen2.5-blue)](https://huggingface.co/Qwen)
[![Falcon3](https://img.shields.io/badge/Model-Falcon3-green)](https://huggingface.co/tiiuae/Falcon3-3B-Instruct)

</div>

---

## 📖 Overview

This project studies efficient small language models for Bangla language understanding.

<p align="justify">
Three Qwen2.5 Instruct models are evaluated under multiple precision settings on three Bangla tasks. Falcon3-3B Instruct is additionally evaluated under the same settings for cross-family validation. The implementation covers model loading, quantization, deterministic prompting, controlled long-context handling, checkpoint-based evaluation, result parsing, efficiency benchmarking, and post-experiment analysis.
</p>

<p align="justify">
The project is motivated by the practical challenge of using capable language models in low-resource languages and environments where memory and computational resources are limited.
</p>

---

## 🤖 Models and Quantization

The experiments use three Qwen2.5 models and Falcon3-3B for cross-family validation.

| Model | Hugging Face Model ID |
|---|---|
| Qwen2.5 0.5B Instruct | `Qwen/Qwen2.5-0.5B-Instruct` |
| Qwen2.5 1.5B Instruct | `Qwen/Qwen2.5-1.5B-Instruct` |
| Qwen2.5 3B Instruct | `Qwen/Qwen2.5-3B-Instruct` |
| Falcon3 3B Instruct | `tiiuae/Falcon3-3B-Instruct` |

Three inference settings are evaluated:

| Setting | Implementation |
|---|---|
| FP16 | Half-precision baseline |
| INT8 | bitsandbytes 8-bit quantization |
| 4-bit NF4 | bitsandbytes NF4 quantization |

The repository uses `int4` internally for the 4-bit NF4 configuration.

<p align="justify">
All precision settings use the same task data, prompts, generation settings, and maximum input length to provide controlled comparisons across quantization levels.
</p>

---

## 📚 Evaluation Tasks

| Task | Labels | Test Examples |
|---|---|---:|
| Stance Classification | Pro-Uprising, Neutral, Anti-Uprising | 982 |
| Natural Language Inference | Entailment, Neutral, Contradiction | 1,253 |
| Fake News Detection | Authentic, Fake | 851 |

<p align="justify">
The NLI data use group-aware splitting so that examples generated from the same premise remain within the same partition, preventing premise leakage across splits.
</p>

<p align="justify">
For fake-news detection, headlines and article content are combined before evaluation, while controlled truncation handles articles exceeding the maximum input length.
</p>

Dataset sources, access information, and preparation details are available in [`data/README.md`](data/README.md).

---

## 🧪 Experimental Setup

<p align="justify">
The complete study contains 36 official evaluations: 27 Qwen2.5 configurations and 9 Falcon3-3B configurations across three tasks and three precision settings.
</p>

| Component | Setting |
|---|---|
| Qwen2.5 model sizes | 0.5B, 1.5B, 3B |
| Cross-family model | Falcon3-3B |
| Precision settings | FP16, INT8, 4-bit NF4 |
| Tasks | Stance, NLI, Fake News |
| Random seed | 42 |
| Maximum input length | 4096 tokens |
| Maximum generated tokens | 8 |
| Sampling | Disabled |
| Primary metric | Macro-F1 |

Generation is deterministic:

```yaml
max_input_tokens: 4096
max_new_tokens: 8
do_sample: false
temperature: null
```

Additional metrics include Accuracy, Macro Precision, Macro Recall, per-class F1, and parse success rate.

---

## 📊 Main Results

<p align="justify">
The results show that compression behavior depends strongly on model capacity and task difficulty, with INT8 consistently preserving more performance than 4-bit NF4 for Qwen2.5-3B.
</p>

| Task | FP16 Macro-F1 | INT8 Macro-F1 | NF4 Macro-F1 | INT8 Retention | NF4 Retention |
|---|---:|---:|---:|---:|---:|
| Stance | 0.435 | 0.434 | 0.397 | 99.9% | 91.3% |
| NLI | 0.728 | 0.700 | 0.563 | 96.2% | 77.3% |
| Fake News | 0.664 | 0.639 | 0.592 | 96.2% | 89.2% |

<p align="justify">
The smallest Qwen2.5 model exhibits class-prediction collapse on some tasks even before quantization, showing that insufficient model capacity can become a limitation before compression itself.
</p>

<p align="center">
  <img
    src="assets/compression_degradation_curves.png"
    alt="Performance degradation under quantization"
    width="95%">
</p>

### Efficiency

The efficiency experiments were conducted on an NVIDIA Tesla T4.

| Precision | Model Footprint | Reduction | Generated Tokens/s |
|---|---:|---:|---:|
| FP16 | 5.748 GB | — | 11.91 |
| INT8 | 3.164 GB | 45.0% | 4.18 |
| 4-bit NF4 | 1.872 GB | 67.4% | 8.92 |

<p align="justify">
INT8 and NF4 substantially reduce the Qwen2.5-3B memory footprint, although these memory savings do not translate directly into faster inference on the tested hardware.
</p>

---

## 📁 Repository Structure

```text
bangla-llm-compression/
│
├── analysis/
│   ├── compression_sensitivity_3b.py    # 3B quantization sensitivity analysis
│   ├── cross_family_comparison.py       # Qwen2.5 vs Falcon3 comparison
│   ├── degradation_curves.py            # performance degradation analysis
│   ├── final_analysis.py                # final experiment analysis
│   ├── generate_figures.py              # paper figure generation
│   ├── performance_memory_scatter.py    # performance-memory trade-off
│   └── retention_heatmap.py             # performance retention visualization
│
├── assets/
│   └── compression_degradation_curves.png
│
├── configs/
│   └── experiment.yaml                  # models and experiment configuration
│
├── data/
│   ├── README.md                        # dataset access and preparation guide
│   ├── prepare_datasets.py              # preprocessing for all three tasks
│   └── create_splits.py                 # deterministic train/dev/test splits
│
├── efficiency/
│   └── benchmark.py                     # memory, latency and throughput benchmark
│
├── evaluation/
│   ├── evaluate.py                      # main evaluation pipeline
│   ├── metrics.py                       # evaluation metrics
│   ├── parsing.py                       # generated-label parsing
│   └── prompts.py                       # task-specific prompts
│
├── models/
│   └── load_model.py                    # FP16, INT8 and NF4 model loading
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

<p align="justify">
Datasets, experiment outputs, generated figures, model checkpoints, and logs are excluded from version control to keep the repository focused on reproducible code.
</p>

---

## 🚀 How to Use This Repository

### 1. Clone the Repository

```bash
git clone https://github.com/sabbir5622r/bangla-llm-compression.git
cd bangla-llm-compression
```

### 2. Install Dependencies

A CUDA-capable GPU is required by the current model-loading implementation.

```bash
pip install -r requirements.txt
```

The main experiment configuration is defined in:

```text
configs/experiment.yaml
```

### 3. Prepare the Datasets

Raw datasets are not redistributed with this repository.

See [`data/README.md`](data/README.md) for dataset sources, access instructions, preparation requirements, and applicable usage conditions.

After obtaining the required datasets, place them according to the instructions in `data/README.md`.

Run preprocessing:

```bash
python data/prepare_datasets.py
```

Create the experimental splits:

```bash
python data/create_splits.py
```

<p align="justify">
The preprocessing and splitting scripts reproduce the task data used by the evaluation pipeline with a fixed random seed of 42.
</p>

### 4. Run an Evaluation

The evaluation function can be imported directly from `evaluation.evaluate`.

```python
from evaluation.evaluate import evaluate

results, metrics = evaluate(
    model_name="qwen2.5-3b-instruct",
    task="nli",
    quantization="int8",
    split="test",
    limit=None,
    checkpoint_every=25,
    resume=True,
)

print(metrics)
```

Available models:

```text
qwen2.5-0.5b-instruct
qwen2.5-1.5b-instruct
qwen2.5-3b-instruct
falcon3-3b-instruct
```

Available tasks:

```text
stance
nli
fake_news
```

Available precision settings:

```text
fp16
int8
int4
```

<p align="justify">
Long evaluations support checkpoint-based recovery, allowing interrupted experiments to resume without repeating examples that have already been successfully processed.
</p>

For a small pipeline test, set `limit` to a small value:

```python
results, metrics = evaluate(
    model_name="qwen2.5-0.5b-instruct",
    task="stance",
    quantization="fp16",
    split="test",
    limit=25,
)
```

Limited runs and complete evaluations use separate output filenames.

### 5. Run the Efficiency Benchmark

Run:

```bash
python efficiency/benchmark.py
```

<p align="justify">
The benchmark measures model footprint, GPU memory usage, latency, examples per second, and generated-token throughput under standardized inference settings.
</p>

<p align="justify">
Reported efficiency results were measured on an NVIDIA Tesla T4, so latency and throughput may differ when reproducing the benchmark on other hardware.
</p>

### 6. Run the Analysis

Post-experiment analysis scripts are available under:

```text
analysis/
```

<p align="justify">
These scripts reproduce the compression, retention, efficiency, and cross-family analyses used to study quantization behavior across the evaluated configurations.
</p>

---

## 📚 Citation

If you use this repository or its experimental framework, please consider citing the associated paper:

```bibtex
@misc{hossen2026quantization,
      title={How Much Can We Compress Small LLMs? Quantization Trade-offs for Low-Resource Bangla Language Understanding}, 
      author={Md Sabbir Hossen and Anichur Rahman and Pabon Shaha and Andrew H. Sung and Md Shohel Rana},
      year={2026},
      eprint={2610.8184772},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.24615}, 
}
```

---

<div align="center">

**Built for reproducible research on efficient LLMs and low-resource language understanding.**  
*If this repository supports your research, consider giving it a ⭐ and citing the paper.*

</div>