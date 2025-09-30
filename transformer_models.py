import os
from typing import Dict, Optional

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

from preprocessing import DatasetSplits

DEFAULT_MAX_TRAIN = 3000
DEFAULT_MAX_EVAL = 2000


def _tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def _maybe_subsample(dataset: Dataset, max_samples: Optional[int], seed: int = 42) -> Dataset:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def _get_env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    return int(value) if value else None


def _get_env_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    return float(value) if value else None


def train_transformer(dataset: DatasetSplits) -> Dict[str, float]:
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    train_dataset = Dataset.from_dict({"text": list(dataset.X_train), "label": list(dataset.y_train)})
    val_dataset = Dataset.from_dict({"text": list(dataset.X_val), "label": list(dataset.y_val)})
    test_dataset = Dataset.from_dict({"text": list(dataset.X_test), "label": list(dataset.y_test)})

    max_train_samples = _get_env_int("MAX_TRAIN_SAMPLES") or DEFAULT_MAX_TRAIN
    max_eval_samples = _get_env_int("MAX_EVAL_SAMPLES") or DEFAULT_MAX_EVAL
    train_dataset = _maybe_subsample(train_dataset, max_train_samples)
    val_dataset = _maybe_subsample(val_dataset, max_eval_samples)
    test_dataset = _maybe_subsample(test_dataset, max_eval_samples)

    print(
        f"Using {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test samples"
    )

    tokenized_train = train_dataset.map(lambda x: _tokenize_function(x, tokenizer), batched=True)
    tokenized_val = val_dataset.map(lambda x: _tokenize_function(x, tokenizer), batched=True)
    tokenized_test = test_dataset.map(lambda x: _tokenize_function(x, tokenizer), batched=True)

    for ds in (tokenized_train, tokenized_val, tokenized_test):
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    val_labels = np.array(tokenized_val['label'])
    test_labels = np.array(tokenized_test['label'])

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(dataset.label_encoder.classes_),
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    num_train_epochs = _get_env_float("NUM_TRAIN_EPOCHS") or 2.0
    per_device_train_batch_size = _get_env_int("TRAIN_BATCH_SIZE") or 8
    per_device_eval_batch_size = _get_env_int("EVAL_BATCH_SIZE") or 8
    learning_rate = _get_env_float("LEARNING_RATE") or 5e-5

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="no",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        warmup_ratio=0.06,
        fp16=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n📊 Transformer Validation Results:\n")
    val_predictions = trainer.predict(tokenized_val)
    val_preds = np.argmax(val_predictions.predictions, axis=1)
    print(classification_report(val_labels, val_preds, target_names=dataset.label_encoder.classes_))

    print("\n📊 Transformer Test Results:\n")
    test_predictions = trainer.predict(tokenized_test)
    test_preds = np.argmax(test_predictions.predictions, axis=1)
    print(classification_report(test_labels, test_preds, target_names=dataset.label_encoder.classes_))

    return {
        "val_accuracy": accuracy_score(val_labels, val_preds),
        "test_accuracy": accuracy_score(test_labels, test_preds),
    }
