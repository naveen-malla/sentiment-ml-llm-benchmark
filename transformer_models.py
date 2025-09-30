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


def _tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def _maybe_subsample(dataset: Dataset, max_samples: Optional[int], seed: int = 42) -> Dataset:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def train_transformer(dataset: DatasetSplits) -> Dict[str, float]:
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    train_dataset = Dataset.from_dict({"text": list(dataset.X_train), "label": list(dataset.y_train)})
    val_dataset = Dataset.from_dict({"text": list(dataset.X_val), "label": list(dataset.y_val)})
    test_dataset = Dataset.from_dict({"text": list(dataset.X_test), "label": list(dataset.y_test)})

    max_train_samples = os.getenv("MAX_TRAIN_SAMPLES")
    max_train_samples = int(max_train_samples) if max_train_samples else None
    train_dataset = _maybe_subsample(train_dataset, max_train_samples)

    tokenized_train = train_dataset.map(lambda x: _tokenize_function(x, tokenizer), batched=True)
    tokenized_val = val_dataset.map(lambda x: _tokenize_function(x, tokenizer), batched=True)
    tokenized_test = test_dataset.map(lambda x: _tokenize_function(x, tokenizer), batched=True)

    for ds in (tokenized_train, tokenized_val, tokenized_test):
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(dataset.label_encoder.classes_),
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="no",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
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
    print(classification_report(dataset.y_val, val_preds, target_names=dataset.label_encoder.classes_))

    print("\n📊 Transformer Test Results:\n")
    test_predictions = trainer.predict(tokenized_test)
    test_preds = np.argmax(test_predictions.predictions, axis=1)
    print(classification_report(dataset.y_test, test_preds, target_names=dataset.label_encoder.classes_))

    return {
        "val_accuracy": accuracy_score(dataset.y_val, val_preds),
        "test_accuracy": accuracy_score(dataset.y_test, test_preds),
    }
