import os
import json

# === YOUR CORRECT CREDENTIALS ===
DATA = {"username":"kumarishr","key":"c431d6ba83996abca7c75de906ff5228"}

# === TARGET: THE SAFE FOLDER ===
SAFE_KAGGLE_DIR = r"C:\MLE_TEMP\.kaggle"
SAFE_KEY_PATH = os.path.join(SAFE_KAGGLE_DIR, "kaggle.json")

def save_key():
    print("🔑 SAVING CORRECT LEGACY KEY")
    print("==========================")
    
    # 1. Create directory if needed
    if not os.path.exists(SAFE_KAGGLE_DIR):
        os.makedirs(SAFE_KAGGLE_DIR)

    # 2. Write the file
    try:
        with open(SAFE_KEY_PATH, "w") as f:
            json.dump(DATA, f)
            
        print(f"✅ Success! Key saved to: {SAFE_KEY_PATH}")
        print(f"   Username: {DATA['username']}")
        print(f"   Key: {DATA['key'][:5]}... (Legacy Format confirmed)")
        print("\n👉 YOU ARE READY. Run 'python final_fix_windows.py' now.")
        
    except Exception as e:
        print(f"❌ Error writing file: {e}")

if __name__ == "__main__":
    save_key()