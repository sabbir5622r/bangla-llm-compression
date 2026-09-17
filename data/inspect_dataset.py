from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def text_stats(series: pd.Series) -> dict:
    s = series.dropna().astype(str)

    if len(s) == 0:
        return {
            "count": 0,
            "min_chars": 0,
            "median_chars": 0,
            "mean_chars": 0,
            "p95_chars": 0,
            "max_chars": 0,
            "mean_words": 0,
            "max_words": 0,
        }

    lengths = s.str.len()
    words = s.str.split().str.len()

    return {
        "count": int(len(s)),
        "min_chars": int(lengths.min()),
        "median_chars": float(lengths.median()),
        "mean_chars": float(lengths.mean()),
        "p95_chars": float(lengths.quantile(0.95)),
        "max_chars": int(lengths.max()),
        "mean_words": float(words.mean()),
        "max_words": int(words.max()),
    }


def inspect_csv(
    path: Path,
    text_columns: list[str],
    label_column: str | None = None,
) -> None:

    print("\n" + "=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    if not path.exists():
        print("ERROR: File not found.")
        return

    df = pd.read_csv(path)

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nExact duplicate rows:")
    print(int(df.duplicated().sum()))

    if label_column and label_column in df.columns:
        print("\nLabel distribution:")
        print(df[label_column].value_counts(dropna=False))

        print("\nLabel proportions:")
        print(
            df[label_column]
            .value_counts(normalize=True, dropna=False)
            .round(4)
        )

    for col in text_columns:

        if col in df.columns:

            print(f"\nText statistics: {col}")
            stats = text_stats(df[col])

            for key, value in stats.items():
                print(f"  {key}: {value}")

    print("\nSample rows:")
    print(df.head(5).to_string(index=False))


def inspect_sentiment(cfg: dict) -> None:

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    ds = cfg["datasets"]["sentiment"]

    path = raw_dir / ds["file"]

    inspect_csv(
        path=path,
        text_columns=[ds["text_column"]],
        label_column=ds["label_column"],
    )


def inspect_fake_news(cfg: dict) -> None:

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    ds = cfg["datasets"]["fake_news"]

    for filename in [
        ds["authentic_file"],
        ds["fake_file"],
    ]:

        path = raw_dir / filename

        inspect_csv(
            path=path,
            text_columns=ds["text_columns"],
            label_column=ds["label_column"],
        )


def inspect_nli(cfg: dict) -> None:

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    ds = cfg["datasets"]["nli"]

    path = raw_dir / ds["file"]

    # For now, inspect every column because we have not
    # yet finalized the exact NLI representation.
    df = pd.read_csv(path)

    print("\n" + "=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    if not path.exists():
        print("ERROR: File not found.")
        return

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nExact duplicate rows:")
    print(int(df.duplicated().sum()))

    print("\nUnique values per column:")

    for col in df.columns:
        print(f"\n--- {col} ---")
        print(f"Unique values: {df[col].nunique(dropna=False)}")

        # Show value distribution for relatively categorical columns.
        if df[col].nunique(dropna=False) <= 20:
            print(df[col].value_counts(dropna=False))

    print("\nText statistics:")

    for col in df.columns:

        if df[col].dtype == "object":

            print(f"\n{col}:")
            stats = text_stats(df[col])

            for key, value in stats.items():
                print(f"  {key}: {value}")

    print("\nSample rows:")
    print(df.head(5).to_string(index=False))


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
    )

    parser.add_argument(
        "--task",
        choices=[
            "all",
            "sentiment",
            "fake_news",
            "nli",
        ],
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