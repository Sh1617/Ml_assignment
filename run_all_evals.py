import subprocess
import sys
import os
from pathlib import Path

SCRIPTS = [
    "eval_siim_isic.py",
    "eval_spooky_author.py", 
    "eval_tps_may_2022.py",
    "eval_text_normalization.py",
    "eval_whale_challenge.py"
]

def create_sample_data():
    """Creates synthetic train.csv files for all datasets"""
    os.makedirs("data", exist_ok=True)
    
    datasets = {
        "siim-isic-melanoma-classification": {
            "cols": ["sex", "age_approx", "target"],
            "n_rows": 1000
        },
        "spooky-author-identification": {
            "cols": ["text", "author"],
            "n_rows": 1000
        },
        "tabular-playground-series-may-2022": {
            "cols": ["feature_1", "feature_2", "feature_3", "target"],
            "n_rows": 1000
        },
        "text-normalization-challenge-english-language": {
            "cols": ["text", "normalized"],
            "n_rows": 500
        },
        "the-icml-2013-whale-challenge-right-whale-redux": {
            "cols": ["feature_1", "feature_2", "whale_id"],
            "n_rows": 800
        }
    }
    
    for dataset, config in datasets.items():
        os.makedirs(f"data/{dataset}", exist_ok=True)
        
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame()
        
        if dataset == "siim-isic-melanoma-classification":
            df["sex"] = np.random.choice(["male", "female"], config["n_rows"])
            df["age_approx"] = np.random.normal(50, 15, config["n_rows"]).clip(0, 100)
            df["target"] = np.random.binomial(1, 0.02, config["n_rows"])
            
        elif dataset == "spooky-author-identification":
            texts = ["This is a sample text for author classification.", 
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning is fascinating and powerful."]
            df["text"] = np.random.choice(texts, config["n_rows"])
            df["author"] = np.random.choice(["EAP", "HPL", "MWS"], config["n_rows"])
            
        elif dataset == "tabular-playground-series-may-2022":
            df["feature_1"] = np.random.normal(0, 1, config["n_rows"])
            df["feature_2"] = np.random.normal(0, 1, config["n_rows"])
            df["feature_3"] = np.random.randint(0, 10, config["n_rows"])
            df["target"] = (df["feature_1"] + df["feature_2"] > 0).astype(int)
            
        elif dataset == "text-normalization-challenge-english-language":
            raw_texts = ["$ 123 dollars", "10.5 %", "Mr. Smith"]
            normalized = [ "$ 123 $", "10.5 %", "Mr. Smith"]
            df["text"] = np.random.choice(raw_texts, config["n_rows"])
            df["normalized"] = np.random.choice(normalized, config["n_rows"])
            
        else:  # whale challenge
            df["feature_1"] = np.random.normal(0, 1, config["n_rows"])
            df["feature_2"] = np.random.normal(0, 1, config["n_rows"])
            df["whale_id"] = np.random.choice([f"whale_{i}" for i in range(10)], config["n_rows"])
        
        df.to_csv(f"data/{dataset}/train.csv", index=False)
        print(f"✅ Created sample data: data/{dataset}/train.csv ({len(df)} rows)")

def main():
    print("🔄 Creating sample data for MLEbench datasets...")
    create_sample_data()
    
    print("\n🚀 Running MLEbench Lite Evaluation Metrics...")
    print("=" * 60)
    
    for script in SCRIPTS:
        try:
            print(f"\n📊 Running {script}...")
            result = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=30)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("❌ ERROR:", result.stderr)
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout: {script}")
        except Exception as e:
            print(f"❌ Error running {script}: {e}")
    
    print("\n🎉 All evaluations complete! Check README table for metrics.")

if __name__ == "__main__":
    main()
