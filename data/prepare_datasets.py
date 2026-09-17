from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


# ============================================================
# VALID LABELS
# ============================================================

SENTIMENT_LABELS = {
    "Positive",
    "Negative",
    "Neutral",
}

FAKE_LABELS = {
    0,
    1,
}

NLI_LABELS = {
    "Entailment",
    "Neutral",
    "Contradiction",
}


# ============================================================
# CONFIG
# ============================================================

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# SENTIMENT DATASET
# ============================================================

def clean_sentiment(cfg: dict) -> pd.DataFrame:

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    out_dir = Path(cfg["paths"]["processed_data_dir"])

    out_dir.mkdir(parents=True, exist_ok=True)

    ds = cfg["datasets"]["sentiment"]

    path = raw_dir / ds["file"]

    if not path.exists():
        raise FileNotFoundError(
            f"Sentiment dataset not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "comment",
        "language",
        "platform",
        "label",
        "emotion",
        "stance",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Sentiment dataset missing columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Initial information
    # --------------------------------------------------------

    before = len(df)

    missing_comments = int(df["comment"].isna().sum())

    # --------------------------------------------------------
    # Remove missing comments
    # --------------------------------------------------------

    df = df.dropna(subset=["comment"]).copy()

    after_missing = len(df)

    # --------------------------------------------------------
    # Normalize platform metadata
    # --------------------------------------------------------

    df["platform"] = df["platform"].replace(
        {
            "Youtube": "YouTube"
        }
    )

    # --------------------------------------------------------
    # Remove exact duplicate rows
    # --------------------------------------------------------

    duplicate_count = int(df.duplicated().sum())

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate sentiment labels
    # --------------------------------------------------------

    found_labels = set(
        df["label"]
        .dropna()
        .unique()
    )

    if not found_labels.issubset(SENTIMENT_LABELS):

        bad_labels = sorted(
            found_labels - SENTIMENT_LABELS
        )

        raise ValueError(
            f"Unexpected sentiment labels: {bad_labels}"
        )

    # --------------------------------------------------------
    # Main task dataset
    #
    # comment -> sentiment label
    #
    # emotion, stance and platform are NOT model inputs.
    # --------------------------------------------------------

    task_df = df[
        [
            "comment",
            "label",
        ]
    ].copy()

    task_df = task_df.rename(
        columns={
            "comment": "text"
        }
    )

    # --------------------------------------------------------
    # Save task dataset
    # --------------------------------------------------------

    task_path = out_dir / "sentiment.csv"

    task_df.to_csv(
        task_path,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save metadata separately
    # --------------------------------------------------------

    metadata_df = df[
        [
            "comment",
            "language",
            "platform",
            "emotion",
            "stance",
            "label",
        ]
    ].copy()

    metadata_path = out_dir / "sentiment_metadata.csv"

    metadata_df.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SENTIMENT PREPROCESSING")
    print("=" * 80)

    print(f"Input file: {path}")
    print(f"Input rows: {before}")
    print(f"Missing comments found: {missing_comments}")
    print(f"After missing-comment removal: {after_missing}")
    print(f"Exact duplicates removed: {duplicate_count}")
    print(f"Final rows: {len(task_df)}")

    print("\nLabel distribution:")
    print(task_df["label"].value_counts())

    print("\nLabel proportions:")
    print(
        task_df["label"]
        .value_counts(normalize=True)
        .round(4)
    )

    print(f"\nSaved task dataset: {task_path}")
    print(f"Saved metadata: {metadata_path}")

    return task_df


# ============================================================
# FAKE-NEWS DATASET
# ============================================================

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
        raise FileNotFoundError(
            f"Fake-news dataset not found: {fake_path}"
        )

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

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    for name, df in [
        ("authentic", authentic),
        ("fake", fake),
    ]:

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} dataset missing columns: "
                f"{sorted(missing)}"
            )

    # --------------------------------------------------------
    # Combine authentic + fake
    # --------------------------------------------------------

    combined = pd.concat(
        [
            authentic,
            fake,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    found_labels = set(
        combined["label"]
        .dropna()
        .unique()
    )

    if not found_labels.issubset(FAKE_LABELS):

        bad_labels = sorted(
            found_labels - FAKE_LABELS
        )

        raise ValueError(
            f"Unexpected fake-news labels: {bad_labels}"
        )

    # --------------------------------------------------------
    # Missing text information
    # --------------------------------------------------------

    missing_headlines = int(
        combined["headline"].isna().sum()
    )

    missing_content = int(
        combined["content"].isna().sum()
    )

    # --------------------------------------------------------
    # Prepare text
    #
    # headline + content -> label
    # --------------------------------------------------------

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
        combined["headline"]
        + "\n\n"
        + combined["content"]
    ).str.strip()

    # --------------------------------------------------------
    # Remove examples where both headline and content are empty
    # --------------------------------------------------------

    empty_text_count = int(
        (combined["text"].str.len() == 0).sum()
    )

    combined = combined[
        combined["text"].str.len() > 0
    ].copy()

    # --------------------------------------------------------
    # Main task dataset
    # --------------------------------------------------------

    task_df = combined[
        [
            "articleID",
            "text",
            "label",
        ]
    ].copy()

    # --------------------------------------------------------
    # Duplicate article IDs
    # --------------------------------------------------------

    duplicate_article_ids = int(
        task_df.duplicated(
            subset=["articleID"]
        ).sum()
    )

    task_df = (
        task_df
        .drop_duplicates(
            subset=["articleID"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save main task dataset
    # --------------------------------------------------------

    task_path = out_dir / "fake_news.csv"

    task_df.to_csv(
        task_path,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Preserve metadata separately
    # --------------------------------------------------------

    metadata_cols = [
        column
        for column in [
            "articleID",
            "domain",
            "date",
            "category",
            "source",
            "relation",
            "F-type",
            "label",
        ]
        if column in combined.columns
    ]

    metadata_path = out_dir / "fake_news_metadata.csv"

    combined[
        metadata_cols
    ].to_csv(
        metadata_path,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FAKE-NEWS PREPROCESSING")
    print("=" * 80)

    print(f"Authentic file: {authentic_path}")
    print(f"Fake file: {fake_path}")

    print(f"\nAuthentic rows: {len(authentic)}")
    print(f"Fake rows: {len(fake)}")

    print(
        "Combined rows: "
        f"{len(authentic) + len(fake)}"
    )

    print(f"Missing headlines: {missing_headlines}")
    print(f"Missing content: {missing_content}")
    print(f"Empty combined texts removed: {empty_text_count}")

    print(
        "Duplicate article IDs removed: "
        f"{duplicate_article_ids}"
    )

    print(f"Final rows: {len(task_df)}")

    print("\nLabel distribution:")
    print(task_df["label"].value_counts())

    print("\nLabel proportions:")
    print(
        task_df["label"]
        .value_counts(normalize=True)
        .round(4)
    )

    print(f"\nSaved task dataset: {task_path}")
    print(f"Saved metadata: {metadata_path}")

    return task_df


# ============================================================
# BANGLA NLI DATASET
# ============================================================

def clean_nli(cfg: dict) -> pd.DataFrame:

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    out_dir = Path(cfg["paths"]["processed_data_dir"])

    out_dir.mkdir(parents=True, exist_ok=True)

    ds = cfg["datasets"]["nli"]

    path = raw_dir / ds["file"]

    if not path.exists():
        raise FileNotFoundError(
            f"NLI dataset not found: {path}"
        )

    # --------------------------------------------------------
    # Load local NLI CSV
    # --------------------------------------------------------

    df = pd.read_csv(path)

    required_columns = {
        "Premise",
        "Entailment",
        "Neutral",
        "Contradiction",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"NLI dataset missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    original_rows = len(df)
    original_duplicates = int(df.duplicated().sum())

    print("\n" + "=" * 80)
    print("BANGLA NLI PREPROCESSING")
    print("=" * 80)

    print(f"Input file: {path}")
    print(f"Original shape: {df.shape}")
    print(f"Original duplicate rows: {original_duplicates}")

    print("\nMissing values:")

    for column in required_columns:
        print(
            f"{column}: "
            f"{int(df[column].isna().sum())}"
        )

    # --------------------------------------------------------
    # Convert wide NLI structure
    #
    # Original:
    #
    # Premise
    # Entailment
    # Neutral
    # Contradiction
    #
    # Becomes:
    #
    # Premise | Hypothesis | label
    #
    # Each original row can therefore create three examples.
    # --------------------------------------------------------

    label_columns = {
        "Entailment": "Entailment",
        "Neutral": "Neutral",
        "Contradiction": "Contradiction",
    }

    records = []

    skipped_missing_premise = 0
    skipped_missing_hypothesis = 0

    # group_id preserves the original row relationship.
    # This will be important when creating train/dev/test splits.
    for original_index, row in df.iterrows():

        premise = row["Premise"]

        # ----------------------------------------------------
        # Validate premise
        # ----------------------------------------------------

        if pd.isna(premise):
            skipped_missing_premise += 1
            continue

        premise = str(premise).strip()

        if not premise:
            skipped_missing_premise += 1
            continue

        # ----------------------------------------------------
        # Generate the three premise-hypothesis pairs
        # ----------------------------------------------------

        for column_name, label_name in label_columns.items():

            hypothesis = row[column_name]

            if pd.isna(hypothesis):
                skipped_missing_hypothesis += 1
                continue

            hypothesis = str(hypothesis).strip()

            if not hypothesis:
                skipped_missing_hypothesis += 1
                continue

            records.append(
                {
                    "group_id": original_index,
                    "Premise": premise,
                    "Hypothesis": hypothesis,
                    "label": label_name,
                }
            )

    # --------------------------------------------------------
    # Create standardized NLI dataframe
    # --------------------------------------------------------

    nli_df = pd.DataFrame(records)

    if nli_df.empty:
        raise ValueError(
            "No valid NLI examples were generated."
        )

    generated_examples = len(nli_df)

    # --------------------------------------------------------
    # Exact duplicate pair detection
    #
    # group_id is intentionally excluded from duplicate
    # detection because it is only a source-row identifier.
    # --------------------------------------------------------

    duplicate_mask = nli_df.duplicated(
        subset=[
            "Premise",
            "Hypothesis",
            "label",
        ]
    )

    duplicate_pairs = int(
        duplicate_mask.sum()
    )

    nli_df = (
        nli_df[
            ~duplicate_mask
        ]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate final labels
    # --------------------------------------------------------

    found_labels = set(
        nli_df["label"].unique()
    )

    if not found_labels.issubset(NLI_LABELS):

        bad_labels = sorted(
            found_labels - NLI_LABELS
        )

        raise ValueError(
            f"Unexpected NLI labels: {bad_labels}"
        )

    # --------------------------------------------------------
    # Save main NLI dataset
    #
    # IMPORTANT:
    # Keep group_id.
    #
    # Later we will split by group_id instead of randomly
    # splitting individual hypothesis rows. This prevents
    # hypotheses from the same original premise appearing
    # across train/dev/test.
    # --------------------------------------------------------

    task_path = out_dir / "nli.csv"

    nli_df.to_csv(
        task_path,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(f"\nOriginal rows: {original_rows}")

    print(
        "Maximum possible generated examples: "
        f"{original_rows * 3}"
    )

    print(
        "Skipped missing/empty premises: "
        f"{skipped_missing_premise}"
    )

    print(
        "Skipped missing/empty hypotheses: "
        f"{skipped_missing_hypothesis}"
    )

    print(
        "Generated valid premise-hypothesis pairs: "
        f"{generated_examples}"
    )

    print(
        "Exact duplicate NLI pairs removed: "
        f"{duplicate_pairs}"
    )

    print(
        f"Final NLI examples: {len(nli_df)}"
    )

    print("\nLabel distribution:")

    print(
        nli_df["label"]
        .value_counts()
    )

    print("\nLabel proportions:")

    print(
        nli_df["label"]
        .value_counts(normalize=True)
        .round(4)
    )

    print("\nSample processed examples:")

    print(
        nli_df.head(6)
        .to_string(index=False)
    )

    print(f"\nSaved task dataset: {task_path}")

    return nli_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare datasets for the "
            "Bangla LLM compression experiments."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to experiment configuration file.",
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
        help="Dataset to prepare.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    cfg = load_config(
        args.config
    )

    # --------------------------------------------------------
    # Prepare requested datasets
    # --------------------------------------------------------

    if args.task in {
        "all",
        "sentiment",
    }:
        clean_sentiment(cfg)

    if args.task in {
        "all",
        "fake_news",
    }:
        clean_fake_news(cfg)

    if args.task in {
        "all",
        "nli",
    }:
        clean_nli(cfg)

    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()