"""Generate dataset statistics and model performance visualisations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing import DatasetSplits, load_and_preprocess_data  # noqa: E402
from traditional_models import train_and_evaluate_classical_models  # noqa: E402
from transformer_models import train_transformer  # noqa: E402

REPORT_DIR = Path("reports")
FIGURES_DIR = REPORT_DIR / "figures"
REPORT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _dataset_stats(dataset: DatasetSplits) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for split_name, texts, labels in [
        ("train", dataset.X_train, dataset.y_train),
        ("val", dataset.X_val, dataset.y_val),
        ("test", dataset.X_test, dataset.y_test),
    ]:
        df = pd.DataFrame({"text": texts, "label": labels})
        df["text_len"] = df["text"].str.len()
        stats[split_name] = {
            "num_samples": float(len(df)),
            "mean_length": float(df["text_len"].mean()),
            "median_length": float(df["text_len"].median()),
            "std_length": float(df["text_len"].std()),
            "min_length": float(df["text_len"].min()),
            "max_length": float(df["text_len"].max()),
        }
    return stats


def _label_distribution(dataset: DatasetSplits) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for split_name, texts, labels in [
        ("Train", dataset.X_train, dataset.y_train),
        ("Validation", dataset.X_val, dataset.y_val),
        ("Test", dataset.X_test, dataset.y_test),
    ]:
        label_counts = pd.Series(labels).value_counts().sort_index()
        for label_idx, count in label_counts.items():
            label_name = dataset.label_encoder.inverse_transform([label_idx])[0]
            rows.append({
                "split": split_name,
                "sentiment": label_name,
                "count": count,
            })
    return pd.DataFrame(rows)


def _length_dataframe(dataset: DatasetSplits) -> pd.DataFrame:
    frames = []
    for split_name, texts in [
        ("Train", dataset.X_train),
        ("Validation", dataset.X_val),
        ("Test", dataset.X_test),
    ]:
        frames.append(pd.DataFrame({"split": split_name, "length": texts.str.len()}))
    return pd.concat(frames, ignore_index=True)


def _model_metrics_frame(classical_results: Dict[str, Dict[str, Dict]], transformer_results: Dict) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_name, split_dict in classical_results.items():
        for split_name, metrics in split_dict.items():
            report = metrics.get("report", {})
            macro = report.get("macro avg", {})
            rows.append(
                {
                    "model": model_name,
                    "split": split_name.capitalize(),
                    "accuracy": metrics.get("accuracy"),
                    "macro_precision": macro.get("precision"),
                    "macro_recall": macro.get("recall"),
                    "macro_f1": macro.get("f1-score"),
                }
            )
    if transformer_results:
        for split_name, label in [
            ("training", "Training"),
            ("validation", "Validation"),
            ("test", "Test"),
        ]:
            metrics = transformer_results.get(split_name, {})
            report = metrics.get("report", {})
            macro = report.get("macro avg", {})
            rows.append(
                {
                    "model": "DistilBERT",
                    "split": label,
                    "accuracy": metrics.get("accuracy"),
                    "macro_precision": macro.get("precision"),
                    "macro_recall": macro.get("recall"),
                    "macro_f1": macro.get("f1-score"),
                }
            )
    return pd.DataFrame(rows)


def _save_json(data: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _plot_label_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="sentiment", y="count", hue="split")
    plt.title("Sentiment distribution by split")
    plt.ylabel("Count")
    plt.xlabel("Sentiment")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_distribution.png", dpi=200)
    plt.close()


def _plot_length_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="split", y="length")
    plt.title("Tweet length distribution")
    plt.ylabel("Characters")
    plt.xlabel("Split")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "text_length_distribution.png", dpi=200)
    plt.close()


def _plot_model_metrics(metrics_df: pd.DataFrame, metric: str, title: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=metrics_df, x="model", y=metric, hue="split")
    plt.title(title)
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close()


def main() -> None:
    dataset = load_and_preprocess_data()

    stats = _dataset_stats(dataset)
    label_df = _label_distribution(dataset)
    length_df = _length_dataframe(dataset)

    classical_results = train_and_evaluate_classical_models(dataset)

    # Ensure reproducible, manageable transformer run
    os.environ.setdefault("MAX_TRAIN_SAMPLES", "3000")
    os.environ.setdefault("MAX_EVAL_SAMPLES", "2000")
    os.environ.setdefault("NUM_TRAIN_EPOCHS", "1")
    os.environ.setdefault("TRAIN_BATCH_SIZE", "16")
    os.environ.setdefault("EVAL_BATCH_SIZE", "32")
    transformer_results = train_transformer(dataset)

    metrics_df = _model_metrics_frame(classical_results, transformer_results)

    _save_json(stats, REPORT_DIR / "dataset_stats.json")
    _save_json(
        {
            "classical": classical_results,
            "transformer": transformer_results,
        },
        REPORT_DIR / "model_metrics.json",
    )

    _plot_label_distribution(label_df)
    _plot_length_distribution(length_df)
    for metric, title, filename in [
        ("accuracy", "Model accuracy by split", "model_accuracy.png"),
        ("macro_f1", "Macro F1-score by split", "model_macro_f1.png"),
        ("macro_precision", "Macro precision by split", "model_macro_precision.png"),
        ("macro_recall", "Macro recall by split", "model_macro_recall.png"),
    ]:
        _plot_model_metrics(metrics_df, metric, title, filename)

    print("Report artefacts saved to:", REPORT_DIR.resolve())
if __name__ == "__main__":
    main()
