import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from metrics import compute_accuracy, print_metrics

DATA_DIR = Path("data/text-normalization-challenge-english-language")

def load_data():
    df = pd.read_csv(DATA_DIR / "train.csv")
    X_train, X_valid, y_train, y_valid = train_test_split(
        df['text'], df['normalized'], test_size=0.2, random_state=42
    )
    return (X_train, y_train), (X_valid, y_valid)

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=100)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train_vec, y_train)
    return model, vectorizer

def main():
    (X_train_text, y_train), (X_valid_text, y_valid) = load_data()
    model, vectorizer = train_model(X_train_text, y_train)
    
    X_valid_vec = vectorizer.transform(X_valid_text)
    y_pred_labels = model.predict(X_valid_vec)
    
    metrics = {"accuracy": compute_accuracy(y_valid.values, y_pred_labels)}
    print_metrics("text-normalization-challenge-english-language", metrics)

if __name__ == "__main__":
    main()
