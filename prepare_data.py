import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

VALID_LABELS: List[str] = ["Positive", "Negative", "Neutral"]


def load_raw_data(train_path: Path, val_path: Path) -> pd.DataFrame:
    columns = ["tweet_id", "entity", "sentiment", "text"]
    df_train = pd.read_csv(train_path, names=columns, header=None)
    df_val = pd.read_csv(val_path, names=columns, header=None)
    df_train["split"] = "train_raw"
    df_val["split"] = "val_raw"
    return pd.concat([df_train, df_val], ignore_index=True)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["sentiment"].isin(VALID_LABELS)].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text", "sentiment"], keep="first")
    return df


def split_by_tweet_id(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_ids = df["tweet_id"].unique()
    train_ids, temp_ids = train_test_split(unique_ids, train_size=train_size, random_state=random_state, shuffle=True)
    relative_val_size = val_size / (1 - train_size)
    val_ids, test_ids = train_test_split(temp_ids, test_size=relative_val_size, random_state=random_state, shuffle=True)
    train_df = df[df["tweet_id"].isin(train_ids)].copy()
    val_df = df[df["tweet_id"].isin(val_ids)].copy()
    test_df = df[df["tweet_id"].isin(test_ids)].copy()

    return train_df, val_df, test_df


def validate_labels(df: pd.DataFrame) -> None:
    conflicts = df.groupby("tweet_id")["sentiment"].nunique()
    problematic_ids = conflicts[conflicts > 1]
    if not problematic_ids.empty:
        joined = ", ".join(map(str, problematic_ids.index[:10]))
        raise ValueError(
            "Found tweet_ids with conflicting sentiment labels. Example ids: "
            f"{joined}. Please resolve before proceeding."
        )


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in {
        "train.csv": train_df,
        "val.csv": val_df,
        "test.csv": test_df,
    }.items():
        split_df.to_csv(output_dir / name, index=False)


def prepare_dataset(
    train_path: Path = Path("data/twitter_training.csv"),
    val_path: Path = Path("data/twitter_validation.csv"),
    output_dir: Path = Path("data/processed"),
    train_size: float = 0.8,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_raw_data(train_path, val_path)
    df = basic_clean(df)
    validate_labels(df)

    train_df, val_df, test_df = split_by_tweet_id(df, train_size, val_size, seed)
    save_splits(train_df, val_df, test_df, output_dir)
    return train_df, val_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cleaned sentiment dataset splits.")
    parser.add_argument("--train", type=Path, default=Path("data/twitter_training.csv"), help="Path to raw training CSV.")
    parser.add_argument("--val", type=Path, default=Path("data/twitter_validation.csv"), help="Path to raw validation CSV.")
    parser.add_argument("--output", type=Path, default=Path("data/processed"), help="Directory to store processed splits.")
    parser.add_argument("--train_size", type=float, default=0.8, help="Proportion of tweet_ids to place in training split.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of tweet_ids to place in validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")

    args = parser.parse_args()

    train_df, val_df, test_df = prepare_dataset(
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    print("✅ Dataset preparation complete")
    print(f"Train/Val/Test sizes: {len(train_df)} / {len(val_df)} / {len(test_df)}")


if __name__ == "__main__":
    main()
