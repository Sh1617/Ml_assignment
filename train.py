import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# === CONFIGURATION ===
# Use the exact ID from your folder structure
COMP_ID = "the-icml-2013-whale-challenge-right-whale-redux"
BASE_DIR = Path("data") / COMP_ID
TRAIN_DIR = BASE_DIR / "train2"
TEST_DIR = BASE_DIR / "test2"
OUTPUT_DIR = Path(f"results/{COMP_ID}/seed_42")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_features(file_path):
    """
    Extracts audio features (MFCCs) from the .aif file.
    Returns a numpy array of features.
    """
    try:
        # Load audio (only first 1 second to speed up processing)
        y, sr = librosa.load(file_path, sr=None, duration=1.0)
        
        if len(y) == 0: return np.zeros(20) # Handle empty audio

        # Calculate MFCCs (texture of sound)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        # Take the mean of each MFCC band over time
        return np.mean(mfcc, axis=1)
    except Exception as e:
        # If file fails, return zeros
        return np.zeros(20)

def main():
    print(f"🚀 Starting Training for {COMP_ID}")
    
    # 1. LOAD THE LABELS (The file you just created!)
    train_csv_path = BASE_DIR / "train.csv"
    if not train_csv_path.exists():
        print("❌ Error: train.csv not found. Did you run Step 1?")
        return
        
    df_train = pd.read_csv(train_csv_path)
    print(f"📄 Loaded train.csv with {len(df_train)} rows.")
    
    # 2. EXTRACT FEATURES FOR TRAINING
    # We will limit to 1000 samples for the 2-hour deadline (Remove [:1000] for full run)
    print("🎧 Processing Training Audio (limiting to 1000 for speed)...")
    
    X_train = []
    y_train = []
    
    # Loop through the first 1000 files
    # Note: df_train['image'] contains the filename like 'train00001_1.aif'
    for _, row in tqdm(df_train.iloc[:1000].iterrows(), total=1000):
        fname = row['image']
        label = row['whale_id']
        path = TRAIN_DIR / fname
        
        if path.exists():
            feats = extract_features(path)
            X_train.append(feats)
            y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # 3. TRAIN MODEL
    print("🧠 Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)
    clf.fit(X_train, y_train)

    # 4. PREDICT ON TEST SET
    # We need the sample_submission to know which files to predict
    sample_sub_path = list(BASE_DIR.glob("sample_*.csv"))[0]
    df_test = pd.read_csv(sample_sub_path)
    print(f"📝 Found test template: {sample_sub_path.name}")
    
    print("⚡ Processing Test Audio...")
    X_test = []
    # Identify the column with filenames (usually first column)
    id_col = df_test.columns[0] 
    
    for fname in tqdm(df_test[id_col]):
        path = TEST_DIR / fname
        if path.exists():
            feats = extract_features(path)
        else:
            feats = np.zeros(20) # Fallback if file missing
        X_test.append(feats)

    # 5. GENERATE SUBMISSION
    probs = clf.predict_proba(np.array(X_test))[:, 1] # Probability of Class 1 (Right Whale)
    
    df_test.iloc[:, 1] = probs
    output_path = OUTPUT_DIR / "submission.csv"
    df_test.to_csv(output_path, index=False)
    
    print(f"🎉 SUCCESS! Submission saved to:\n{output_path}")
    print("👉 You are ready for the final grading step!")

if __name__ == "__main__":
    main()