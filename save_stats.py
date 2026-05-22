# save_stats.py
import pandas as pd
import numpy as np
import librosa
import json
import os

# Ensure this matches your actual CSV name
csv_path = "train_manifest.csv"
df = pd.read_csv(csv_path)

means, stds = [], []

print(f"Calculating stats for {len(df)} files...")

for idx, path in enumerate(df['path']):
    # Ensure the path uses the correct slashes for Windows
    clean_path = path.replace('/', '\\')
    
    if not os.path.exists(clean_path):
        print(f"File not found: {clean_path}")
        continue
        
    try:
        # Load audio
        audio, _ = librosa.load(clean_path, sr=16000, duration=4)
        
        # Extract features
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        means.append(np.mean(mel_db))
        stds.append(np.std(mel_db))
        
        if idx % 50 == 0:
            print(f"Processed {idx} files...")
            
    except Exception as e:
        print(f"Error processing {clean_path}: {e}")

# Save the results
if means:
    stats = {"mean": float(np.mean(means)), "std": float(np.mean(stds))}
    with open("stats.json", "w") as f:
        json.dump(stats, f)
    print(f"\nSUCCESS! Stats saved to stats.json: {stats}")
else:
    print("\nError: No files were processed. Check your CSV paths.")