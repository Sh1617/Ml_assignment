import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from metrics import compute_multiclass_logloss, print_metrics

DATA_DIR = Path("data/spooky-author-identification")

def load_data():
    df = pd.read_csv(DATA_DIR / "train.csv")
    le = LabelEncoder()
    y = le.fit_transform(df['author'])
    X_train, X_valid, y_train, y_valid = train_test_split(
        df['text'], y, test_size=0.2, random_state=42, stratify=y
    )
    return (X_train, y_train), (X_valid, y_valid), le

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    model.fit(X_train_vec, y_train)
    return model, vectorizer

def main():
    (X_train_text, y_train), (X_valid_text, y_valid), le = load_data()
    model, vectorizer = train_model(X_train_text, y_train)
    
    X_valid_vec = vectorizer.transform(X_valid_text)
    y_pred_proba = model.predict_proba(X_valid_vec)
    
    metrics = {"log_loss": compute_multiclass_logloss(y_valid, y_pred_proba)}
    print_metrics("spooky-author-identification", metrics)

if __name__ == "__main__":
    main()
