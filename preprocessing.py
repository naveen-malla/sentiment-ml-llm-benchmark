import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder

from prepare_data import prepare_dataset

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


@dataclass
class DatasetSplits:
    X_train: pd.Series
    y_train: pd.Series
    X_val: pd.Series
    y_val: pd.Series
    X_test: pd.Series
    y_test: pd.Series
    label_encoder: LabelEncoder


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


def _ensure_processed_dataset(processed_dir: Path) -> None:
    required_files = [processed_dir / name for name in ("train.csv", "val.csv", "test.csv")]
    if all(path.exists() for path in required_files):
        return
    prepare_dataset(output_dir=processed_dir)


def _load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_columns = {'tweet_id', 'entity', 'sentiment', 'text'}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Split file {path} missing columns: {sorted(missing)}")
    df = df.rename(columns={'sentiment': 'label'})
    df['text'] = df['text'].astype(str)
    df['clean_text'] = df['text'].apply(clean_text)
    return df[['tweet_id', 'entity', 'label', 'clean_text']]


def load_and_preprocess_data(processed_dir: str = "data/processed") -> DatasetSplits:
    processed_path = Path(processed_dir)
    _ensure_processed_dataset(processed_path)

    splits: Dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        splits[name] = _load_split(processed_path / f"{name}.csv")

    label_encoder = LabelEncoder()
    splits['train']['label_encoded'] = label_encoder.fit_transform(splits['train']['label'])
    for split_name in ('val', 'test'):
        splits[split_name]['label_encoded'] = label_encoder.transform(splits[split_name]['label'])

    return DatasetSplits(
        X_train=splits['train']['clean_text'],
        y_train=splits['train']['label_encoded'],
        X_val=splits['val']['clean_text'],
        y_val=splits['val']['label_encoded'],
        X_test=splits['test']['clean_text'],
        y_test=splits['test']['label_encoded'],
        label_encoder=label_encoder,
    )


if __name__ == "__main__":
    load_and_preprocess_data()
