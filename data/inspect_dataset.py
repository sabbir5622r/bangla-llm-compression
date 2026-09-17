from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def text_stats(series: pd.Series) -> dict:
    text = series.dropna().astype(str)

    if text.empty:
        return {}

    chars = text.str.len()
    words = text.str.split().str.len()

    return {
        "count": len(text),
        "min_chars": chars.min(),
        "median_chars": chars.median(),
        "mean_chars": chars.mean(),
        "p95_chars": chars.quantile(0.95),
        "max_chars": chars.max(),
        "mean_words": words.mean(),
        "max_words": words.max(),
    }


def print_text_stats(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue

        stats = text_stats(df[col])

        print(f"\nText statistics: {col}")

        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")


def inspect_sentiment(cfg: dict) -> None:
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    ds = cfg["datasets"]["sentiment"]
    path = raw_dir / ds["file"]

    if not path.exists():
        raise FileNotFoundError(f"Sentiment dataset not found: {path}")

    df = pd.read_csv(path)

    print("\nSentiment")
    print(f"File: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nMissing values:")
    print(df.isna().sum())

    print(f"\nExact duplicates: {df.duplicated().sum()}")

    print("\nLabel distribution:")
    print(df[ds["label_column"]].value_counts(dropna=False))

    print("\nLabel proportions:")
    print(
        df[ds["label_column"]]
        .value_counts(normalize=True, dropna=False)
        .round(4)
    )

    if "platform" in df.columns:
        print("\nPlatform distribution:")
        print(df["platform"].value_counts(dropna=False))

    print_text_stats(df, [ds["text_column"]])

    print("\nSample:")
    print(df.head(3).to_string(index=False))


def inspect_fake_news(cfg: dict) -> None:
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    ds = cfg["datasets"]["fake_news"]

    files = [
        ("Authentic", ds["authentic_file"]),
        ("Fake", ds["fake_file"]),
    ]

    for name, filename in files:
        path = raw_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"{name} dataset not found: {path}")

        df = pd.read_csv(path)

        print(f"\n{name} News")
        print(f"File: {path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        print("\nMissing values:")
        print(df.isna().sum())

        print(f"\nExact duplicates: {df.duplicated().sum()}")

        print("\nLabel distribution:")
        print(df[ds["label_column"]].value_counts(dropna=False))

        print_text_stats(df, ds["text_columns"])

        if "F-type" in df.columns:
            print("\nF-type distribution:")
            print(df["F-type"].value_counts(dropna=False))

        print("\nSample:")
        print(df.head(3).to_string(index=False))


def inspect_nli(cfg: dict) -> None:
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    ds = cfg["datasets"]["nli"]
    path = raw_dir / ds["file"]

    if not path.exists():
        raise FileNotFoundError(f"NLI dataset not found: {path}")

    df = pd.read_csv(path)

    print("\nNLI")
    print(f"File: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nMissing values:")
    print(df.isna().sum())

    print(f"\nExact duplicates: {df.duplicated().sum()}")

    columns = [
        "Premise",
        "Entailment",
        "Neutral",
        "Contradiction",
    ]

    print_text_stats(df, columns)

    print("\nSample:")
    print(df.head(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
    )

    parser.add_argument(
        "--task",
        choices=["all", "sentiment", "fake_news", "nli"],
        default="all",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.task in {"all", "sentiment"}:
        inspect_sentiment(cfg)

    if args.task in {"all", "fake_news"}:
        inspect_fake_news(cfg)

    if args.task in {"all", "nli"}:
        inspect_nli(cfg)


if __name__ == "__main__":
    main()