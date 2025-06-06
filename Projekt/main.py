import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv("data/diabetes_text_descriptions.csv")
X = data["text"]
y = data["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

logreg = LogisticRegression(max_iter=1000)
nb = MultinomialNB()
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)

logreg.fit(X_train_tfidf, y_train)
nb.fit(X_train_tfidf, y_train)
mlp.fit(X_train_tfidf, y_train)

y_pred_logreg = logreg.predict(X_test_tfidf)
y_pred_nb = nb.predict(X_test_tfidf)
y_pred_mlp = mlp.predict(X_test_tfidf)

def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-score": round(f1_score(y_true, y_pred, zero_division=0), 4)
    }

results = [
    evaluate_model("TF-IDF + Logistic Regression", y_test, y_pred_logreg),
    evaluate_model("TF-IDF + Naive Bayes", y_test, y_pred_nb),
    evaluate_model("TF-IDF + MLP Neural Network", y_test, y_pred_mlp),
]

print("\n=== Porównanie modeli ===")
for res in results:
    print(f"\nModel: {res['Model']}")
    print(f"  Accuracy:  {res['Accuracy']}")
    print(f"  Precision: {res['Precision']}")
    print(f"  Recall:    {res['Recall']}")
    print(f"  F1-score:  {res['F1-score']}")
