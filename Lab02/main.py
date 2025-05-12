import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# nltk.download('punkt')

train_path = "data/train.tsv.gz"
dev_path = "data/dev_in.tsv"
dev_labels_path = "data/dev_expected.tsv"
test_path = "data/test_in.tsv"

train = pd.read_csv(train_path, sep="\t", header=None, names=["label", "text"], on_bad_lines='skip')
dev = pd.read_csv(dev_path, sep="\t", header=None, names=["text"], on_bad_lines='skip')
dev_labels = pd.read_csv(dev_labels_path, sep="\t", header=None, names=["label"], on_bad_lines='skip')

train["tokens"] = train["text"].apply(lambda x: word_tokenize(str(x).lower()))
dev["tokens"] = dev["text"].apply(lambda x: word_tokenize(str(x).lower()))

tagged_train = [TaggedDocument(words=row["tokens"], tags=[str(i)]) for i, row in train.iterrows()]

doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4)
doc2vec_model.build_vocab(tagged_train)

EPOCHS = 3
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS} - Doc2Vec")
    doc2vec_model.train(tagged_train, total_examples=doc2vec_model.corpus_count, epochs=1)
    doc2vec_model.alpha -= 0.002
    doc2vec_model.min_alpha = doc2vec_model.alpha

X_train = np.array([doc2vec_model.dv[str(i)] for i in range(len(train))])
y_train = train["label"].astype(int).values

X_dev = np.array([doc2vec_model.infer_vector(tokens, alpha=0.025, epochs=10) for tokens in dev["tokens"]])
y_dev = dev_labels["label"].astype(int).values

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining Neural Network...")
model.fit(X_train, y_train, epochs=10, batch_size=32)

y_pred_prob = model.predict(X_dev).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n--- Evaluation Results ---")
print(f"Accuracy: {accuracy_score(y_dev, y_pred)}")
print("\nClassification Report:\n", classification_report(y_dev, y_pred))
