# ✅ main.py
from preprocessing import load_and_preprocess_data
from traditional_models import train_and_evaluate_classical_models
from transformer_models import train_transformer


def main():
    print("\n🔄 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(
        "data/twitter_training.csv",
        "data/twitter_validation.csv"
    )

    print("\n📊 Running Traditional ML Models...")
    train_and_evaluate_classical_models(X_train, X_test, y_train, y_test, label_encoder)

    print("\n🤖 Running Transformer (BERT) Model...")
    train_transformer(X_train, X_test, y_train, y_test, label_encoder)


if __name__ == "__main__":
    main()
