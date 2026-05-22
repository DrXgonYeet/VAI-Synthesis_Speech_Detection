import os
import pandas as pd
import numpy as np
import librosa
import json

# Get the directory where the script is located (the project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv("train_manifest.csv")
means, stds = [], []

print("Calculating dataset statistics...")

for path in df['path']:
    # Combine project root with the relative path from the CSV
    full_path = os.path.join(PROJECT_ROOT, path)
    
    if not os.path.exists(full_path):
        print(f"Skipping (not found): {full_path}")
        continue
        
    try:
        # Load audio
        audio, _ = librosa.load(full_path, sr=16000, duration=4)
        
        # Extract features
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        means.append(np.mean(mel_db))
        stds.append(np.std(mel_db))
        
    except Exception as e:
        print(f"Skipping (error): {full_path} - {e}")

# Save the results
if means:
    stats = {"mean": float(np.mean(means)), "std": float(np.mean(stds))}
    with open("stats.json", "w") as f:
        json.dump(stats, f)
    print(f"\nSUCCESS! Stats saved to stats.json: {stats}")
else:
    print("\nError: No files were processed.")