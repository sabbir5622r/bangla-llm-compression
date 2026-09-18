from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def load_config(path="configs/experiment.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_stratified(df, seed):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df["label"],
    )

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"],
    )

    return train_df, dev_df, test_df


def split_nli(df, seed):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    train_idx, temp_idx = next(
        splitter.split(
            df,
            groups=df["group_id"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed,
    )

    dev_idx, test_idx = next(
        splitter.split(
            temp_df,
            groups=temp_df["group_id"],
        )
    )

    dev_df = temp_df.iloc[dev_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    return train_df, dev_df, test_df


def save_splits(df, task, out_dir, seed):
    if task == "nli":
        train_df, dev_df, test_df = split_nli(df, seed)
    else:
        train_df, dev_df, test_df = split_stratified(df, seed)

    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        task_dir / "train.csv",
        index=False,
        encoding="utf-8",
    )

    dev_df.to_csv(
        task_dir / "dev.csv",
        index=False,
        encoding="utf-8",
    )

    test_df.to_csv(
        task_dir / "test.csv",
        index=False,
        encoding="utf-8",
    )

    print(f"\n{task}")
    print(f"Total: {len(df)}")
    print(f"Train: {len(train_df)}")
    print(f"Dev: {len(dev_df)}")
    print(f"Test: {len(test_df)}")

    print("\nTrain labels:")
    print(train_df["label"].value_counts())

    print("\nDev labels:")
    print(dev_df["label"].value_counts())

    print("\nTest labels:")
    print(test_df["label"].value_counts())

    if task == "nli":
        train_groups = set(train_df["group_id"])
        dev_groups = set(dev_df["group_id"])
        test_groups = set(test_df["group_id"])

        assert train_groups.isdisjoint(dev_groups)
        assert train_groups.isdisjoint(test_groups)
        assert dev_groups.isdisjoint(test_groups)

        print("\nNLI group leakage: None")


def main():
    cfg = load_config()

    seed = cfg["project"]["seed"]

    processed_dir = Path(
        cfg["paths"]["processed_data_dir"]
    )

    out_dir = processed_dir / "splits"

    files = {
        "sentiment": "sentiment.csv",
        "nli": "nli.csv",
        "fake_news": "fake_news.csv",
    }

    for task, filename in files.items():
        path = processed_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {path}"
            )

        df = pd.read_csv(path)

        save_splits(
            df,
            task,
            out_dir,
            seed,
        )


if __name__ == "__main__":
    main()