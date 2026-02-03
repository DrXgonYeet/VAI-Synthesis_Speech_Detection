import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf

# CONFIGURATION
DATA_ROOT = Path("data")
ASV_PATH = DATA_ROOT / "ASVspoof2019_LA"
CV_PATH = DATA_ROOT / "common_voice"
TARGET_SR = 16000

def process_asvspoof():
    print("Processing ASVspoof 2019 LA...")
    protocol_path = ASV_PATH / "ASVspoof2019_LA_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    audio_dir = ASV_PATH / "ASVspoof2019_LA_train/flac"
    
    data = []
    # Read ASVspoof protocol file
    # Format: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
    # KEY: 'bonafide' (Human) or 'spoof' (AI)
    with open(protocol_path, 'r') as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        parts = line.strip().split()
        filename = parts[1]
        label_str = parts[-1] # 'bonafide' or 'spoof'
        
        # 0 = Human, 1 = AI
        label = 0 if label_str == 'bonafide' else 1
        
        file_path = audio_dir / f"{filename}.flac"
        if file_path.exists():
            data.append({
                "path": str(file_path),
                "label": label,
                "source": "asvspoof"
            })
    return data

def process_common_voice(lang_code):
    print(f"Processing Common Voice ({lang_code})...")
    lang_dir = CV_PATH / lang_code
    # Assuming standard CV structure: clips/ folder and train.tsv
    clips_dir = lang_dir / "clips"
    tsv_path = lang_dir / "train.tsv"
    
    if not tsv_path.exists():
        print(f"Skipping {lang_code}: train.tsv not found")
        return []

    df = pd.read_csv(tsv_path, sep='\t')
    data = []
    
    # Process only first 2000 files per language to prevent imbalance
    # (Since we don't have Indian AI voices, too much Real data will bias the model)
    limit = 2000 
    
    for _, row in tqdm(df.head(limit).iterrows(), total=min(len(df), limit)):
        mp3_name = row['path']
        src_path = clips_dir / mp3_name
        
        # We must convert MP3 to WAV 16kHz for consistency
        wav_name = mp3_name.replace(".mp3", ".wav")
        dest_path = clips_dir / wav_name
        
        if not dest_path.exists() and src_path.exists():
            try:
                # Convert using librosa (slow but works) or ffmpeg
                y, sr = librosa.load(src_path, sr=TARGET_SR)
                sf.write(dest_path, y, TARGET_SR)
            except Exception as e:
                continue
        
        if dest_path.exists():
            data.append({
                "path": str(dest_path),
                "label": 0, # Common Voice is always Human (0)
                "source": f"cv_{lang_code}"
            })
            
    return data

# RUN
all_data = []
all_data.extend(process_asvspoof())

# Add your languages
for lang in ["hi", "ta", "ml", "te"]:
    all_data.extend(process_common_voice(lang))

# Save unified dataset
df = pd.DataFrame(all_data)
df.to_csv("train_manifest.csv", index=False)
print(f"Saved manifest with {len(df)} files to train_manifest.csv")
print(df['label'].value_counts())