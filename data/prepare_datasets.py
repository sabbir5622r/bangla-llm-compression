from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


SENTIMENT_LABELS = {"Positive", "Negative", "Neutral"}
FAKE_LABELS = {0, 1}
NLI_LABELS = {"Entailment", "Neutral", "Contradiction"}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_sentiment(cfg: dict) -> pd.DataFrame:
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    out_dir = Path(cfg["paths"]["processed_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = cfg["datasets"]["sentiment"]
    path = raw_dir / ds["file"]

    if not path.exists():
        raise FileNotFoundError(f"Sentiment dataset not found: {path}")

    df = pd.read_csv(path)

    required = {"comment", "language", "platform", "label", "emotion", "stance"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Sentiment dataset missing columns: {sorted(missing)}")

    total_rows = len(df)
    missing_comments = df["comment"].isna().sum()

    df = df.dropna(subset=["comment"]).copy()
    df["platform"] = df["platform"].replace({"Youtube": "YouTube"})

    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)

    found_labels = set(df["label"].dropna().unique())

    if not found_labels.issubset(SENTIMENT_LABELS):
        bad_labels = sorted(found_labels - SENTIMENT_LABELS)
        raise ValueError(f"Unexpected sentiment labels: {bad_labels}")

    task_df = df[["comment", "label"]].copy()
    task_df = task_df.rename(columns={"comment": "text"})

    task_path = out_dir / "sentiment.csv"
    task_df.to_csv(task_path, index=False, encoding="utf-8")

    # Keep metadata separate from model input.
    metadata_df = df[
        ["comment", "language", "platform", "emotion", "stance", "label"]
    ].copy()

    metadata_path = out_dir / "sentiment_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8")

    print("\nSentiment")
    print(f"Input rows: {total_rows}")
    print(f"Missing comments removed: {missing_comments}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Final rows: {len(task_df)}")

    print("\nLabel distribution:")
    print(task_df["label"].value_counts())

    print(f"\nSaved: {task_path}")

    return task_df


def clean_fake_news(cfg: dict) -> pd.DataFrame:
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    out_dir = Path(cfg["paths"]["processed_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = cfg["datasets"]["fake_news"]

    authentic_path = raw_dir / ds["authentic_file"]
    fake_path = raw_dir / ds["fake_file"]

    if not authentic_path.exists():
        raise FileNotFoundError(
            f"Authentic-news dataset not found: {authentic_path}"
        )

    if not fake_path.exists():
        raise FileNotFoundError(f"Fake-news dataset not found: {fake_path}")

    authentic = pd.read_csv(authentic_path)
    fake = pd.read_csv(fake_path)

    required = {
        "articleID",
        "domain",
        "date",
        "category",
        "source",
        "relation",
        "headline",
        "content",
        "label",
    }

    for name, df in [("authentic", authentic), ("fake", fake)]:
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} dataset missing columns: {sorted(missing)}"
            )

    combined = pd.concat([authentic, fake], ignore_index=True)

    found_labels = set(combined["label"].dropna().unique())

    if not found_labels.issubset(FAKE_LABELS):
        bad_labels = sorted(found_labels - FAKE_LABELS)
        raise ValueError(f"Unexpected fake-news labels: {bad_labels}")

    missing_headlines = combined["headline"].isna().sum()
    missing_content = combined["content"].isna().sum()

    combined["headline"] = (
        combined["headline"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    combined["content"] = (
        combined["content"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    combined["text"] = (
        combined["headline"] + "\n\n" + combined["content"]
    ).str.strip()

    empty_text_count = (combined["text"].str.len() == 0).sum()
    combined = combined[combined["text"].str.len() > 0].copy()

    task_df = combined[["articleID", "text", "label"]].copy()

    duplicate_ids = task_df.duplicated(subset=["articleID"]).sum()

    task_df = (
        task_df
        .drop_duplicates(subset=["articleID"])
        .reset_index(drop=True)
    )

    task_path = out_dir / "fake_news.csv"
    task_df.to_csv(task_path, index=False, encoding="utf-8")

   
    metadata_cols = [
        col
        for col in [
            "articleID",
            "domain",
            "date",
            "category",
            "source",
            "relation",
            "F-type",
            "label",
        ]
        if col in combined.columns
    ]

    metadata_path = out_dir / "fake_news_metadata.csv"

    combined[metadata_cols].to_csv(
        metadata_path,
        index=False,
        encoding="utf-8",
    )

    print("\nFake News")
    print(f"Authentic rows: {len(authentic)}")
    print(f"Fake rows: {len(fake)}")
    print(f"Missing headlines: {missing_headlines}")
    print(f"Missing content: {missing_content}")
    print(f"Empty texts removed: {empty_text_count}")
    print(f"Duplicate article IDs removed: {duplicate_ids}")
    print(f"Final rows: {len(task_df)}")

    print("\nLabel distribution:")
    print(task_df["label"].value_counts())

    print(f"\nSaved: {task_path}")

    return task_df


def clean_nli(cfg: dict) -> pd.DataFrame:
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    out_dir = Path(cfg["paths"]["processed_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = cfg["datasets"]["nli"]
    path = raw_dir / ds["file"]

    if not path.exists():
        raise FileNotFoundError(f"NLI dataset not found: {path}")

    df = pd.read_csv(path)

    required = {
        "Premise",
        "Entailment",
        "Neutral",
        "Contradiction",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"NLI dataset missing columns: {sorted(missing)}"
        )

    original_rows = len(df)
    original_duplicates = df.duplicated().sum()

    label_columns = {
        "Entailment": "Entailment",
        "Neutral": "Neutral",
        "Contradiction": "Contradiction",
    }

    records = []
    skipped_premises = 0
    skipped_hypotheses = 0

    for row_id, row in df.iterrows():
        premise = row["Premise"]

        if pd.isna(premise) or not str(premise).strip():
            skipped_premises += 1
            continue

        premise = str(premise).strip()

        for column, label in label_columns.items():
            hypothesis = row[column]

            if pd.isna(hypothesis) or not str(hypothesis).strip():
                skipped_hypotheses += 1
                continue

            records.append(
                {
                    "group_id": row_id,
                    "Premise": premise,
                    "Hypothesis": str(hypothesis).strip(),
                    "label": label,
                }
            )

    nli_df = pd.DataFrame(records)

    if nli_df.empty:
        raise ValueError("No valid NLI examples were generated.")

    generated_examples = len(nli_df)

    duplicate_mask = nli_df.duplicated(
        subset=["Premise", "Hypothesis", "label"]
    )

    duplicate_pairs = duplicate_mask.sum()

    nli_df = nli_df[~duplicate_mask].reset_index(drop=True)

    found_labels = set(nli_df["label"].unique())

    if not found_labels.issubset(NLI_LABELS):
        bad_labels = sorted(found_labels - NLI_LABELS)
        raise ValueError(f"Unexpected NLI labels: {bad_labels}")

    
    task_path = out_dir / "nli.csv"
    nli_df.to_csv(task_path, index=False, encoding="utf-8")

    print("\nNLI")
    print(f"Original rows: {original_rows}")
    print(f"Original duplicate rows: {original_duplicates}")
    print(f"Skipped premises: {skipped_premises}")
    print(f"Skipped hypotheses: {skipped_hypotheses}")
    print(f"Generated examples: {generated_examples}")
    print(f"Duplicate pairs removed: {duplicate_pairs}")
    print(f"Final rows: {len(nli_df)}")

    print("\nLabel distribution:")
    print(nli_df["label"].value_counts())

    print(f"\nSaved: {task_path}")

    return nli_df


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
        clean_sentiment(cfg)

    if args.task in {"all", "fake_news"}:
        clean_fake_news(cfg)

    if args.task in {"all", "nli"}:
        clean_nli(cfg)


if __name__ == "__main__":
    main()