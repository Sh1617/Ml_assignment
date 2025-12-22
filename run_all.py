import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Attempt to import librosa for audio; handle gracefully if missing (though you have it)
try:
    import librosa
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("⚠️ Librosa not found. Whale challenge might fail.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
BASE_DIR = Path("data")
RESULTS_DIR = Path("results")
SEED = 42

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1. SIIM-ISIC Melanoma (Image -> Metadata)
# ---------------------------------------------------------
def solve_melanoma():
    comp_id = "siim-isic-melanoma-classification"
    print(f"\n🔹 Processing: {comp_id}")
    
    data_dir = BASE_DIR / comp_id
    out_dir = RESULTS_DIR / comp_id / f"seed_{SEED}"
    ensure_dir(out_dir)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sub = pd.read_csv(data_dir / "sample_submission.csv")

    # Simple Metadata Preprocessing
    def process(df):
        df['sex_f'] = (df['sex'] == 'male').astype(int)
        df['age_approx_f'] = df['age_approx'].fillna(0)
        return df[['sex_f', 'age_approx_f']].fillna(0)

    X_train = process(train)
    y_train = train['target']
    X_test = process(test)

    print("   Training Random Forest...")
    model = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=SEED)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    
    sub['target'] = preds
    sub.to_csv(out_dir / "submission.csv", index=False)
    print(f"   ✅ Saved to {out_dir}/submission.csv")

# ---------------------------------------------------------
# 2. Spooky Author Identification (Text)
# ---------------------------------------------------------
def solve_spooky():
    comp_id = "spooky-author-identification"
    print(f"\n🔹 Processing: {comp_id}")

    data_dir = BASE_DIR / comp_id
    out_dir = RESULTS_DIR / comp_id / f"seed_{SEED}"
    ensure_dir(out_dir)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sub = pd.read_csv(data_dir / "sample_submission.csv")

    print("   Vectorizing Text...")
    vec = TfidfVectorizer(max_features=3000, stop_words='english')
    X_train = vec.fit_transform(train['text'])
    X_test = vec.transform(test['text'])
    
    le = LabelEncoder()
    y_train = le.fit_transform(train['author']) # EAP, HPL, MWS

    print("   Training Random Forest...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=SEED)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)
    
    # Save (Columns must match EAP, HPL, MWS order)
    sub['EAP'] = preds[:, 0]
    sub['HPL'] = preds[:, 1]
    sub['MWS'] = preds[:, 2]
    
    sub.to_csv(out_dir / "submission.csv", index=False)
    print(f"   ✅ Saved to {out_dir}/submission.csv")

# ---------------------------------------------------------
# 3. Tabular Playground May 2022 (Tabular)
# ---------------------------------------------------------
def solve_tps():
    comp_id = "tabular-playground-series-may-2022"
    print(f"\n🔹 Processing: {comp_id}")

    data_dir = BASE_DIR / comp_id
    out_dir = RESULTS_DIR / comp_id / f"seed_{SEED}"
    ensure_dir(out_dir)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sub = pd.read_csv(data_dir / "sample_submission.csv")

    # Drop ID and target, handle f_27 string column by dropping it for speed
    X_train = train.drop(columns=['id', 'target', 'f_27'], errors='ignore').fillna(0)
    y_train = train['target']
    X_test = test.drop(columns=['id', 'f_27'], errors='ignore').fillna(0)

    print("   Training Random Forest...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=SEED)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]

    sub['target'] = preds
    sub.to_csv(out_dir / "submission.csv", index=False)
    print(f"   ✅ Saved to {out_dir}/submission.csv")

# ---------------------------------------------------------
# 4. Text Normalization (Text -> Baseline)
# ---------------------------------------------------------
def solve_text_norm():
    comp_id = "text-normalization-challenge-english-language"
    print(f"\n🔹 Processing: {comp_id}")

    data_dir = BASE_DIR / comp_id
    out_dir = RESULTS_DIR / comp_id / f"seed_{SEED}"
    ensure_dir(out_dir)

    # For Lite benchmark, we create a valid submission even if simple
    try:
        test = pd.read_csv(data_dir / "test.csv")
        sub = pd.read_csv(data_dir / "sample_submission.csv")
    except FileNotFoundError:
        print("   ⚠️ Files missing for Text Norm. Skipping.")
        return

    print("   Generating Baseline Submission...")
    # Baseline: Assume 'after' is same as 'before' (no normalization)
    # This guarantees a valid submission file structure
    sub['after'] = test['before'] 
    
    sub.to_csv(out_dir / "submission.csv", index=False)
    print(f"   ✅ Saved to {out_dir}/submission.csv")

# ---------------------------------------------------------
# 5. Whale Challenge (Audio -> Spectrogram features)
# ---------------------------------------------------------
def extract_whale_features(file_path):
    try:
        # Load only 1 second to keep it fast
        y, sr = librosa.load(file_path, sr=None, duration=1.0)
        if len(y) == 0: return np.zeros(20)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        return np.mean(mfcc, axis=1)
    except:
        return np.zeros(20)

def solve_whale():
    comp_id = "the-icml-2013-whale-challenge-right-whale-redux"
    print(f"\n🔹 Processing: {comp_id}")

    data_dir = BASE_DIR / comp_id
    out_dir = RESULTS_DIR / comp_id / f"seed_{SEED}"
    ensure_dir(out_dir)

    train_dir = data_dir / "train2"
    test_dir = data_dir / "test2"
    train_csv_path = data_dir / "train.csv"

    # Check if we have the generated train.csv
    if not train_csv_path.exists():
        print("   ❌ train.csv missing! Please run the label generator first.")
        return

    df_train = pd.read_csv(train_csv_path)
    
    # Locate sample submission for Test IDs
    # Case insensitive search for sample_submission
    sub_files = list(data_dir.glob("sample_*.csv"))
    if not sub_files:
        print("   ❌ sample_submission.csv missing!")
        return
    sub_file = sub_files[0]
    df_test = pd.read_csv(sub_file)

    print("   🎧 extracting Audio Features (Train)...")
    X_train = []
    y_train = []
    
    # Limit to 1000 for execution speed
    LIMIT = 1000
    for _, row in tqdm(df_train.iloc[:LIMIT].iterrows(), total=LIMIT):
        path = train_dir / row['image']
        if path.exists():
            X_train.append(extract_whale_features(path))
            y_train.append(row['whale_id'])

    print("   Training Random Forest...")
    model = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=SEED)
    model.fit(X_train, y_train)

    print("   🎧 Extracting Audio Features (Test)...")
    X_test = []
    id_col = df_test.columns[0]
    
    for fname in tqdm(df_test[id_col]):
        path = test_dir / fname
        if path.exists():
            X_test.append(extract_whale_features(path))
        else:
            X_test.append(np.zeros(20))

    preds = model.predict_proba(X_test)[:, 1]
    df_test.iloc[:, 1] = preds
    
    df_test.to_csv(out_dir / "submission.csv", index=False)
    print(f"   ✅ Saved to {out_dir}/submission.csv")

# ---------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------
def main():
    print("🚀 STARTING ONE-SHOT RUN FOR ALL 5 COMPETITIONS")
    print("="*50)
    
    solve_melanoma()
    solve_spooky()
    solve_tps()
    solve_text_norm()
    if HAS_AUDIO:
        solve_whale()
    else:
        print("Skipping Whale (No Audio Lib)")

    print("\n" + "="*50)
    print("🎉 DONE! All submissions generated.")
    print("👉 Now run 'python grade_final.py' to see your table.")

if __name__ == "__main__":
    main()