from pathlib import Path

import pandas as pd
import torch

from evaluation.metrics import compute_metrics
from evaluation.parsing import parse_output
from evaluation.prompts import build_prompt
from models.load_model import load_config, load_model


FAKE_NEWS_LABELS = {
    1: "Authentic",
    0: "Fake",
}


def load_dataset(cfg, task):
    processed_dir = Path(cfg["paths"]["processed_data_dir"])

    if task == "sentiment":
        path = processed_dir / "sentiment.csv"
    elif task == "nli":
        path = processed_dir / "nli.csv"
    elif task == "fake_news":
        path = processed_dir / "fake_news.csv"
    else:
        raise ValueError(f"Unknown task: {task}")

    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    df = pd.read_csv(path)

    if task == "fake_news":
        df["label"] = df["label"].map(FAKE_NEWS_LABELS)

        if df["label"].isna().any():
            raise ValueError("Unexpected fake-news labels.")

    return df


def generate_prediction(model, tokenizer, prompt, cfg):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    max_input_tokens = cfg["evaluation"]["max_input_tokens"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=cfg["evaluation"]["max_new_tokens"],
            do_sample=False,
        )

    generated = output[0, input_length:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


def evaluate(
    model_name,
    task,
    quantization="fp16",
    limit=None,
):
    cfg = load_config()

    df = load_dataset(cfg, task)

    if limit is not None:
        df = df.head(limit).copy()

    model, tokenizer = load_model(
        model_name,
        quantization=quantization,
        cfg=cfg,
    )

    results = []

    for index, row in df.iterrows():
        example = row.to_dict()

        prompt = build_prompt(task, example)

        raw_output = generate_prediction(
            model,
            tokenizer,
            prompt,
            cfg,
        )

        predicted_label, parse_success = parse_output(
            raw_output,
            task,
        )

        results.append(
            {
                "example_id": index,
                "true_label": row["label"],
                "predicted_label": predicted_label,
                "raw_output": raw_output,
                "parse_success": parse_success,
            }
        )

    results_df = pd.DataFrame(results)

    valid_df = results_df[results_df["parse_success"]].copy()

    metrics = None

    if not valid_df.empty:
        metrics = compute_metrics(
            valid_df["true_label"].tolist(),
            valid_df["predicted_label"].tolist(),
            task,
        )

    return results_df, metrics