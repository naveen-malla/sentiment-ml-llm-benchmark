# ✅ main.py
from preprocessing import load_and_preprocess_data
from traditional_models import train_and_evaluate_classical_models
from transformer_models import train_transformer


def main():
    print("\n🔄 Loading and preprocessing data...")
    dataset = load_and_preprocess_data()

    print("\n📊 Running Traditional ML Models...")
    train_and_evaluate_classical_models(dataset)

    print("\n🤖 Running Transformer (BERT) Model...")
    transformer_metrics = train_transformer(dataset)
    if transformer_metrics:
        val_acc = transformer_metrics.get("validation", {}).get("accuracy")
        test_acc = transformer_metrics.get("test", {}).get("accuracy")
        print(f"\n✅ Transformer accuracy — val: {val_acc:.3f}, test: {test_acc:.3f}")


if __name__ == "__main__":
    main()
