import gc
import statistics
import time
from pathlib import Path

import pandas as pd
import torch

from models.load_model import load_config, load_model


cfg = load_config()

QWEN_MODELS = [
    "qwen2.5-0.5b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-3b-instruct",
]

FALCON_MODELS = [
    "falcon3-3b-instruct",
]

QUANTIZATIONS = [
    "fp16",
    "int8",
    "int4",
]

WARMUP_RUNS = 3
MEASURED_RUNS = 20
MAX_NEW_TOKENS = 8
TARGET_INPUT_TOKENS = 512

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def build_fixed_input(tokenizer):
    sentence = (
        "বাংলাদেশ একটি দক্ষিণ এশীয় দেশ এবং বাংলা ভাষা "
        "এ অঞ্চলের একটি গুরুত্বপূর্ণ ভাষা। "
    )

    text = sentence

    while len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    ) < 600:
        text += sentence

    messages = [
        {
            "role": "user",
            "content": text,
        }
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=TARGET_INPUT_TOKENS,
    )

    return inputs


def benchmark_configuration(model_name, quantization):
    clear_gpu()

    print("\n" + "=" * 70)
    print(model_name, "|", quantization.upper())
    print("=" * 70)

    torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()

    model, tokenizer = load_model(
        model_name,
        quantization=quantization,
        cfg=cfg,
    )

    torch.cuda.synchronize()

    load_time = time.perf_counter() - load_start

    model_memory_gb = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    model_footprint_gb = (
        model.get_memory_footprint()
        / 1024**3
    )

    inputs = build_fixed_input(tokenizer)

    input_tokens = inputs["input_ids"].shape[1]

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    for _ in range(WARMUP_RUNS):
        with torch.inference_mode():
            _ = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    latencies = []
    generated_token_counts = []

    for _ in range(MEASURED_RUNS):
        torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        generated_tokens = (
            output.shape[1]
            - input_tokens
        )

        latencies.append(elapsed)
        generated_token_counts.append(
            generated_tokens
        )

    peak_memory_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    mean_latency = statistics.mean(
        latencies
    )

    median_latency = statistics.median(
        latencies
    )

    total_generated = sum(
        generated_token_counts
    )

    total_time = sum(
        latencies
    )

    tokens_per_second = (
        total_generated
        / total_time
    )

    examples_per_second = (
        MEASURED_RUNS
        / total_time
    )

    result = {
        "model": model_name,
        "quantization": quantization,
        "gpu": torch.cuda.get_device_name(0),
        "input_tokens": input_tokens,
        "max_new_tokens": MAX_NEW_TOKENS,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "load_time_sec": load_time,
        "model_gpu_memory_gb": model_memory_gb,
        "model_footprint_gb": model_footprint_gb,
        "peak_inference_memory_gb": peak_memory_gb,
        "mean_latency_sec": mean_latency,
        "median_latency_sec": median_latency,
        "examples_per_second": examples_per_second,
        "generated_tokens_per_second": tokens_per_second,
    }

    del output
    del inputs
    del tokenizer
    del model

    clear_gpu()

    return result


def run_benchmark(models):
    results = []

    for model_name in models:
        for quantization in QUANTIZATIONS:
            result = benchmark_configuration(
                model_name,
                quantization,
            )

            results.append(result)

            print("\nResult:")

            for key, value in result.items():
                print(f"{key}: {value}")

    return pd.DataFrame(results)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for efficiency benchmarking."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("PyTorch:", torch.__version__)

    print("\nRunning Qwen2.5 benchmarks")

    qwen_results = run_benchmark(
        QWEN_MODELS
    )

    qwen_path = (
        OUTPUT_DIR
        / "efficiency_benchmark.csv"
    )

    qwen_results.to_csv(
        qwen_path,
        index=False,
    )

    print("\nSaved:", qwen_path)

    print("\nRunning Falcon3 benchmarks")

    falcon_results = run_benchmark(
        FALCON_MODELS
    )

    falcon_path = (
        OUTPUT_DIR
        / "falcon3_efficiency_benchmark.csv"
    )

    falcon_results.to_csv(
        falcon_path,
        index=False,
    )

    print("\nSaved:", falcon_path)

    print("\nQwen2.5 results")

    print(
        qwen_results[
            [
                "model",
                "quantization",
                "model_gpu_memory_gb",
                "model_footprint_gb",
                "peak_inference_memory_gb",
                "mean_latency_sec",
                "generated_tokens_per_second",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nFalcon3 results")

    print(
        falcon_results[
            [
                "model",
                "quantization",
                "model_gpu_memory_gb",
                "model_footprint_gb",
                "peak_inference_memory_gb",
                "mean_latency_sec",
                "generated_tokens_per_second",
            ]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()