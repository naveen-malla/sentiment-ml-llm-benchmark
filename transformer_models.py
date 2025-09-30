# ✅ transformer_model.py
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.metrics import classification_report

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def train_transformer(X_train, X_test, y_train, y_test, label_encoder):
    model_name = "distilbert-base-uncased"  # Smaller, faster model
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    train_dataset = Dataset.from_dict({"text": list(X_train), "label": list(y_train)})
    test_dataset = Dataset.from_dict({"text": list(X_test), "label": list(y_test)})

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",  # Disable evaluation during training
        num_train_epochs=1,
        per_device_train_batch_size=8,   # Small but not too small
        per_device_eval_batch_size=8,
        logging_steps=10,
        save_strategy="no",
        max_steps=50,  # Absolute minimum training steps
        warmup_steps=0,   # No warmup
        learning_rate=5e-5,  # Higher learning rate for faster convergence
        fp16=False,
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    preds = torch.argmax(torch.tensor(predictions.predictions), axis=1)
    print("\n📊 Transformer Model Results:\n")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))
