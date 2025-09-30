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
    train_transformer(dataset)


if __name__ == "__main__":
    main()
