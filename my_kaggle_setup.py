import os
import json
import sys

def setup_kaggle():
    print("🔑 KAGGLE AUTHENTICATION SETUP")
    print("==============================")
    
    # 1. Get User Input
    print("Please enter your Kaggle details (found in 'Create New Token' on Kaggle):")
    username = input("Enter your Kaggle Username: ").strip()
    key = input("Enter your Kaggle Key (the long string): ").strip()
    
    if not username or not key:
        print("❌ Error: Username and Key are required.")
        return

    # 2. Define the Path
    # Windows: C:\Users\<Name>\.kaggle\kaggle.json
    home = os.path.expanduser("~")
    kaggle_dir = os.path.join(home, ".kaggle")
    kaggle_file = os.path.join(kaggle_dir, "kaggle.json")

    # 3. Create Folder and Write File
    try:
        if not os.path.exists(kaggle_dir):
            os.makedirs(kaggle_dir)
            print(f"   Created directory: {kaggle_dir}")
        
        data = {"username": username, "key": key}
        
        with open(kaggle_file, "w") as f:
            json.dump(data, f)
        
        print(f"✅ Successfully created: {kaggle_file}")
        
    except Exception as e:
        print(f"❌ Error creating file: {e}")
        return

    # 4. Verify Connection
    print("\n🕵️ Verifying connection...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("✅ SUCCESS! Your computer can now talk to Kaggle.")
        print("👉 You can now run 'python final_fix.py' to finish grading.")
        
    except ImportError:
        print("⚠️ Kaggle library not installed. Installing now...")
        os.system(f"{sys.executable} -m pip install kaggle")
        print("Please run this script again to verify.")
        
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("   Did you type your username/key correctly?")
        print("   Did you accept the competition rules on the website?")

if __name__ == "__main__":
    setup_kaggle()