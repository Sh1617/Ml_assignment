import subprocess
import os
import sys
import json
from pathlib import Path

# === CONFIGURATION ===
SAFE_ROOT = r"C:\MLE_TEMP"
VENV_PYTHON = os.path.join(SAFE_ROOT, "env", "Scripts", "python.exe")
VENV_PIP = os.path.join(SAFE_ROOT, "env", "Scripts", "pip.exe")
MLEBENCH_CMD = os.path.join(SAFE_ROOT, "env", "Scripts", "mlebench.exe")

# Override Environment variables to force everything into C:\MLE_TEMP
env = os.environ.copy()
env["USERPROFILE"] = SAFE_ROOT
env["LOCALAPPDATA"] = SAFE_ROOT
env["APPDATA"] = SAFE_ROOT
env["KAGGLE_CONFIG_DIR"] = os.path.join(SAFE_ROOT, ".kaggle")
env["PYTHONUTF8"] = "1"

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("mlebench_grade_reports")
OUTPUT_DIR.mkdir(exist_ok=True)

COMPETITIONS = [
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "the-icml-2013-whale-challenge-right-whale-redux"
]

def run_fix_and_grade():
    print("🔧 APPLYING FINAL WINDOWS FIX")
    print("============================")

    # 1. UNINSTALL pywin32 (The cause of the crash)
    print("   Removing conflicting Windows libraries...")
    try:
        subprocess.run([VENV_PIP, "uninstall", "-y", "pywin32", "pypiwin32"], capture_output=True)
        print("   ✅ Libraries removed. Tool will now obey safe paths.")
    except Exception as e:
        print(f"   ⚠️ Warning: {e}")

    print("\n🚀 STARTING FINAL GRADING")
    print("=========================")
    
    summary = []

    for comp_id in COMPETITIONS:
        print(f"\n🔹 PROCESSING: {comp_id}")
        
        # 2. PREPARE
        print("   [1/2] Downloading Data...")
        subprocess.run([MLEBENCH_CMD, "prepare", "-c", comp_id], env=env)

        # 3. GRADE
        print("   [2/2] Grading Submission...")
        sub_path = RESULTS_DIR / comp_id / "seed_42" / "submission.csv"
        
        if not sub_path.exists():
            print(f"   ❌ Missing file: {sub_path}")
            continue
            
        cmd = [MLEBENCH_CMD, "grade-sample", str(sub_path.absolute()), comp_id]
        
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
            
            output = p.stdout
            if "{" in output and "}" in output:
                json_str = output[output.find("{"):output.rfind("}")+1]
                data = json.loads(json_str)
                
                with open(OUTPUT_DIR / f"{comp_id}.json", "w") as f:
                    json.dump(data, f, indent=4)
                
                score = data.get("score", "N/A")
                medal = "YES" if data.get("any_medal") else "No"
                print(f"   ✅ SUCCESS! Score: {score} | Medal: {medal}")
                summary.append({"Competition": comp_id, "Score": score, "Medal": medal})
            else:
                print("   ❌ Grading Failed.")
                if p.stderr: print(f"   STDERR: {p.stderr[:200]}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Final Table
    print("\n" + "="*60)
    print(f"{'COMPETITION':<50} | {'SCORE':<8} | {'MEDAL'}")
    print("-" * 60)
    for item in summary:
        print(f"{item['Competition']:<50} | {str(item['Score']):<8} | {item['Medal']}")
    print("="*60)

if __name__ == "__main__":
    run_fix_and_grade()