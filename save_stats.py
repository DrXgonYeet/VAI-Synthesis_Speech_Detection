# save_stats.py
import pandas as pd
import numpy as np
import librosa
import json

df = pd.read_csv("train_manifest.csv")
means, stds = [], []

print("Calculating dataset statistics (this may take a moment)...")
for path in df['path']:
    audio, _ = librosa.load(path, sr=16000, duration=4)
    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    means.append(np.mean(mel_db))
    stds.append(np.std(mel_db))

stats = {"mean": float(np.mean(means)), "std": float(np.mean(stds))}
with open("stats.json", "w") as f:
    json.dump(stats, f)
print(f"Stats saved to stats.json: {stats}")