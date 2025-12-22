import os
import pandas as pd

# Define paths based on your image
base_dir = "data/the-icml-2013-whale-challenge-right-whale-redux"
train_dir = os.path.join(base_dir, "train2")
output_csv = os.path.join(base_dir, "train.csv")

def generate_csv():
    print(f"📂 Scanning: {train_dir}")
    
    if not os.path.exists(train_dir):
        print("❌ Error: 'train2' folder not found. Check your path.")
        return

    files = os.listdir(train_dir)
    data = []
    
    print(f"   Found {len(files)} files. Processing...")

    for filename in files:
        # Expected format example: "train00001_1.aif" (where 1 is the label)
        # We split by '_' and then take the first character of the last part
        try:
            if "_" in filename:
                parts = filename.split('_')
                # Last part should be like "1.aif" or "0.aif"
                label_part = parts[-1]
                label = int(label_part.split('.')[0])
                
                data.append({'image': filename, 'whale_id': label})
        except:
            continue

    if len(data) > 0:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"✅ Success! Generated train.csv with {len(df)} rows.")
        print(f"📍 Saved to: {output_csv}")
    else:
        print("❌ Could not extract labels. Filenames might be different than expected.")

if __name__ == "__main__":
    generate_csv()