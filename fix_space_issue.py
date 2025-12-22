import os
import subprocess
import sys
import shutil
import urllib.request
import zipfile

# === CONFIGURATION ===
SAFE_ROOT = r"C:\MLE_TEMP"
VENV_PATH = os.path.join(SAFE_ROOT, "env")
PYTHON_EXE = os.path.join(VENV_PATH, "Scripts", "python.exe")
PIP_EXE = os.path.join(VENV_PATH, "Scripts", "pip.exe")
SOURCE_DIR = os.path.join(SAFE_ROOT, "mle-bench-main")
ZIP_PATH = os.path.join(SAFE_ROOT, "mle-bench.zip")
REPO_URL = "https://github.com/openai/mle-bench/archive/refs/heads/main.zip"

def run_setup():
    print("🛡️ STARTING MASTER SETUP (No Git Required)")
    print(f"   Target: {SAFE_ROOT}")
    print("==========================================")

    # 1. Create Safe Directory
    if not os.path.exists(SAFE_ROOT):
        os.makedirs(SAFE_ROOT)
        print(f"✅ Created {SAFE_ROOT}")

    # 2. Create Virtual Environment
    if not os.path.exists(PYTHON_EXE):
        print(f"📦 Creating Python environment...")
        subprocess.run([sys.executable, "-m", "venv", VENV_PATH], check=True)
    else:
        print("✅ Environment already exists.")

    # 3. Upgrade PIP (Fixes older version issues)
    print("⬆️  Upgrading pip...")
    subprocess.run([PYTHON_EXE, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    # 4. Install Standard Libraries
    print("⬇️  Installing dependencies (pandas, kaggle, etc)...")
    pkgs = ["kaggle", "pandas", "scikit-learn", "librosa", "numpy", "tqdm", "setuptools"]
    subprocess.run([PIP_EXE, "install"] + pkgs, check=True)

    # 5. Download MLE-BENCH Manually
    if not os.path.exists(SOURCE_DIR):
        print("🌍 Downloading MLE-Bench source code...")
        try:
            urllib.request.urlretrieve(REPO_URL, ZIP_PATH)
            print("   Extracting zip file...")
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(SAFE_ROOT)
            print("✅ Download & Extraction complete.")
        except Exception as e:
            print(f"❌ Error downloading/extracting: {e}")
            return
    else:
        print("✅ Source code already present.")

    # 6. Install MLE-BENCH from Source
    print("🛠️  Installing MLE-Bench...")
    subprocess.run([PIP_EXE, "install", SOURCE_DIR], check=True)

    # 7. Move Kaggle Key (Crucial Step)
    safe_kaggle_dir = os.path.join(SAFE_ROOT, ".kaggle")
    if not os.path.exists(safe_kaggle_dir):
        os.makedirs(safe_kaggle_dir)
    
    old_key = os.path.expanduser("~/.kaggle/kaggle.json")
    safe_key = os.path.join(safe_kaggle_dir, "kaggle.json")
    
    if os.path.exists(old_key):
        shutil.copy(old_key, safe_key)
        print(f"🔑 Copied kaggle.json to: {safe_key}")
    else:
        print("⚠️  Warning: Could not find your kaggle.json! You might need to paste it manually.")

    print("\n✅ SETUP COMPLETE.")
    print("👉 Now run: 'python final_grade_safe.py'")

if __name__ == "__main__":
    run_setup()