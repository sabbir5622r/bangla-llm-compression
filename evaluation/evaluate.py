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


def load_dataset(
    cfg,
    task,
    data_dir=None,
    split=None,
):
    if data_dir is None:
        processed_dir = Path(
            cfg["paths"]["processed_data_dir"]
        )
    else:
        processed_dir = Path(data_dir)

    filenames = {
        "stance": "stance.csv",
        "nli": "nli.csv",
        "fake_news": "fake_news.csv",
    }

    if task not in filenames:
        raise ValueError(
            f"Unknown task: {task}"
        )

    if split is None:
        path = (
            processed_dir
            / filenames[task]
        )
    else:
        if split not in {
            "train",
            "dev",
            "test",
        }:
            raise ValueError(
                f"Unknown split: {split}"
            )

        path = (
            processed_dir
            / "splits"
            / task
            / f"{split}.csv"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if task == "fake_news":
        df["label"] = df["label"].map(
            FAKE_NEWS_LABELS
        )

        if df["label"].isna().any():
            raise ValueError(
                "Unexpected fake-news labels."
            )

    return df


def format_chat(
    tokenizer,
    prompt,
):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def fit_prompt(
    tokenizer,
    task,
    example,
    max_input_tokens,
):
    prompt = build_prompt(
        task,
        example,
    )

    text = format_chat(
        tokenizer,
        prompt,
    )

    original_input_tokens = len(
        tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
    )

    if (
        original_input_tokens
        <= max_input_tokens
    ):
        return (
            text,
            original_input_tokens,
            original_input_tokens,
            False,
        )

    if task != "fake_news":
        raise ValueError(
            f"{task} input exceeds "
            f"max_input_tokens="
            f"{max_input_tokens}"
        )

    empty_example = example.copy()
    empty_example["text"] = ""

    empty_prompt = build_prompt(
        task,
        empty_example,
    )

    empty_text = format_chat(
        tokenizer,
        empty_prompt,
    )

    prompt_tokens = len(
        tokenizer(
            empty_text,
            add_special_tokens=False,
        )["input_ids"]
    )

    article_budget = (
        max_input_tokens
        - prompt_tokens
    )

    if article_budget <= 0:
        raise ValueError(
            "Prompt instructions exceed "
            "max_input_tokens."
        )

    article_tokens = tokenizer(
        str(example["text"]),
        add_special_tokens=False,
    )["input_ids"]

    truncated_text = tokenizer.decode(
        article_tokens[:article_budget],
        skip_special_tokens=True,
    )

    truncated_example = example.copy()
    truncated_example["text"] = truncated_text

    prompt = build_prompt(
        task,
        truncated_example,
    )

    text = format_chat(
        tokenizer,
        prompt,
    )

    used_input_tokens = len(
        tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
    )

    while (
        used_input_tokens
        > max_input_tokens
    ):
        article_tokens = (
            article_tokens[:-1]
        )

        truncated_example["text"] = (
            tokenizer.decode(
                article_tokens,
                skip_special_tokens=True,
            )
        )

        prompt = build_prompt(
            task,
            truncated_example,
        )

        text = format_chat(
            tokenizer,
            prompt,
        )

        used_input_tokens = len(
            tokenizer(
                text,
                add_special_tokens=False,
            )["input_ids"]
        )

    return (
        text,
        original_input_tokens,
        used_input_tokens,
        True,
    )


def generate_prediction(
    model,
    tokenizer,
    text,
    cfg,
):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    input_length = (
        inputs["input_ids"].shape[1]
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=cfg[
                "evaluation"
            ]["max_new_tokens"],
            do_sample=False,
        )

    generated = output[
        0,
        input_length:,
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


def get_result_path(
    cfg,
    model_name,
    task,
    quantization,
    split,
):
    result_dir = Path(
        cfg["paths"]["raw_result_dir"]
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_model_name = (
        model_name
        .replace("/", "_")
        .replace("\\", "_")
    )

    split_name = (
        split
        if split is not None
        else "full"
    )

    filename = (
        f"{safe_model_name}_"
        f"{quantization}_"
        f"{task}_"
        f"{split_name}.csv"
    )

    return result_dir / filename


def evaluate(
    model_name,
    task,
    quantization="fp16",
    limit=None,
    data_dir=None,
    split=None,
    checkpoint_every=25,
    resume=True,
):
    cfg = load_config()

    df = load_dataset(
        cfg,
        task,
        data_dir=data_dir,
        split=split,
    )

    if limit is not None:
        df = df.head(
            limit
        ).copy()

    result_path = get_result_path(
        cfg,
        model_name,
        task,
        quantization,
        split,
    )

    results = []
    completed_ids = set()

    if resume and result_path.exists():
        checkpoint_df = pd.read_csv(
            result_path
        )

        results = checkpoint_df.to_dict(
            "records"
        )

        completed_ids = set(
            checkpoint_df[
                "example_id"
            ].tolist()
        )

        print(
            f"Resuming from "
            f"{len(completed_ids)} "
            f"completed examples."
        )

    remaining = (
        len(df)
        - len(completed_ids)
    )

    print(
        f"Total examples: {len(df)}"
    )
    print(
        f"Remaining examples: {remaining}"
    )
    print(
        f"Checkpoint: {result_path}"
    )

    if remaining > 0:
        model, tokenizer = load_model(
            model_name,
            quantization=quantization,
            cfg=cfg,
        )

        new_since_save = 0

        for index, row in df.iterrows():
            if index in completed_ids:
                continue

            example = row.to_dict()

            (
                text,
                original_input_tokens,
                used_input_tokens,
                was_truncated,
            ) = fit_prompt(
                tokenizer,
                task,
                example,
                cfg["evaluation"][
                    "max_input_tokens"
                ],
            )

            raw_output = (
                generate_prediction(
                    model,
                    tokenizer,
                    text,
                    cfg,
                )
            )

            (
                predicted_label,
                parse_success,
            ) = parse_output(
                raw_output,
                task,
            )

            results.append(
                {
                    "example_id": index,
                    "true_label": (
                        row["label"]
                    ),
                    "predicted_label": (
                        predicted_label
                    ),
                    "raw_output": (
                        raw_output
                    ),
                    "parse_success": (
                        parse_success
                    ),
                    "original_input_tokens": (
                        original_input_tokens
                    ),
                    "used_input_tokens": (
                        used_input_tokens
                    ),
                    "was_truncated": (
                        was_truncated
                    ),
                }
            )

            completed_ids.add(index)
            new_since_save += 1

            print(
                f"{len(completed_ids)}/"
                f"{len(df)} | "
                f"true: {row['label']} | "
                f"pred: {predicted_label}"
            )

            if (
                new_since_save
                >= checkpoint_every
            ):
                pd.DataFrame(
                    results
                ).to_csv(
                    result_path,
                    index=False,
                    encoding="utf-8",
                )

                print(
                    f"Checkpoint saved: "
                    f"{len(completed_ids)}/"
                    f"{len(df)}"
                )

                new_since_save = 0

    results_df = pd.DataFrame(
        results
    )

    if not results_df.empty:
        results_df = (
            results_df
            .drop_duplicates(
                subset=["example_id"],
                keep="last",
            )
            .sort_values(
                "example_id"
            )
            .reset_index(
                drop=True
            )
        )

        results_df.to_csv(
            result_path,
            index=False,
            encoding="utf-8",
        )

    valid_df = results_df[
        results_df["parse_success"]
    ].copy()

    metrics = None

    if not valid_df.empty:
        metrics = compute_metrics(
            valid_df[
                "true_label"
            ].tolist(),
            valid_df[
                "predicted_label"
            ].tolist(),
            task,
        )

    print(
        f"\nResults saved: "
        f"{result_path}"
    )

    print(
        f"Completed: "
        f"{len(results_df)}/"
        f"{len(df)}"
    )

    print(
        f"Invalid outputs: "
        f"{len(results_df) - len(valid_df)}"
    )

    return (
        results_df,
        metrics,
    )