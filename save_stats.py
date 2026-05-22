import os
import pandas as pd
import numpy as np
import librosa
import json

# Get the directory where the script is located (the project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv("train_manifest.csv")
means, stds = [], []

for path in df['path']:
    # Combine project root with the relative path from the CSV
    # This works regardless of where the project is cloned on someone's computer
    full_path = os.path.join(PROJECT_ROOT, path)
    
    if not os.path.exists(full_path):
        continue
        
    try:
        audio, _ = librosa.load(full_path, sr=16000, duration=4)
        # ... rest of your processing ...