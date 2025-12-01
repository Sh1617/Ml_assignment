import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from metrics import compute_multiclass_logloss, print_metrics

DATA_DIR = Path("data/the-icml-2013-whale-challenge-right-whale-redux")

def load_data():
    df = pd.read_csv(DATA_DIR / "train.csv")
    le = LabelEncoder()
    y = le.fit_transform(df['whale_id'])
    features = [col for col in df.columns if col not in ['whale_id', 'image']]
    if len(features) == 0:
        features = ['feature_1', 'feature_2']
    X = df[features].fillna(0)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return (X_train, y_train), (X_valid, y_valid), le

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    return model

def main():
    (X_train, y_train), (X_valid, y_valid), le = load_data()
    model = train_model(X_train, y_train)
    y_pred_proba = model.predict_proba(X_valid)
    
    metrics = {"log_loss": compute_multiclass_logloss(y_valid, y_pred_proba)}
    print_metrics("the-icml-2013-whale-challenge-right-whale-redux", metrics)

if __name__ == "__main__":
    main()
