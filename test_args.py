from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=2
)

print("✅ Success!")
