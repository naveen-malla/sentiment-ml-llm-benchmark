from typing import Dict

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import DatasetSplits


def _evaluate_split(clf: Pipeline, X, y, label_names, split_name: str) -> Dict[str, float]:
    preds = clf.predict(X)
    accuracy = accuracy_score(y, preds)
    print(f"\n📊 {split_name} results:\n")
    print(classification_report(y, preds, target_names=label_names))
    return {"accuracy": accuracy}


def train_and_evaluate_classical_models(dataset: DatasetSplits) -> Dict[str, Dict[str, float]]:
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        clf = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', model)
        ])
        clf.fit(dataset.X_train, dataset.y_train)

        split_metrics: Dict[str, float] = {}
        for split_name, (X, y) in {
            "Validation": (dataset.X_val, dataset.y_val),
            "Test": (dataset.X_test, dataset.y_test),
        }.items():
            metrics = _evaluate_split(clf, X, y, dataset.label_encoder.classes_, split_name)
            split_metrics[f"{split_name.lower()}_accuracy"] = metrics['accuracy']

        results[name] = split_metrics

    return results
