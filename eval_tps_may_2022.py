import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from metrics import compute_binary_auc, print_metrics

DATA_DIR = Path("data/tabular-playground-series-may-2022")

def load_data():
    df = pd.read_csv(DATA_DIR / "train.csv")
    features = [col for col in df.columns if col != 'target']
    X = df[features].fillna(0)
    y = df['target']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, y_train), (X_valid, y_valid)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    return model

def main():
    (X_train, y_train), (X_valid, y_valid) = load_data()
    model = train_model(X_train, y_train)
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    
    metrics = {"roc_auc": compute_binary_auc(y_valid.values, y_pred_proba)}
    print_metrics("tabular-playground-series-may-2022", metrics)

if __name__ == "__main__":
    main()
