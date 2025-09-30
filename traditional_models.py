# ✅ traditional_models.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def train_and_evaluate_classical_models(X_train, X_test, y_train, y_test, label_encoder):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC()
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        clf = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', model)
        ])
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        print(f"\n📊 Results for {name}:\n")
        print(classification_report(y_test, preds, target_names=label_encoder.classes_))
