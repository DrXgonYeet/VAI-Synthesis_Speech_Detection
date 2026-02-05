import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf
import random

# CONFIGURATION
DATA_ROOT = Path("data")
ASV_PATH = DATA_ROOT / "ASVspoof2019_LA"
CV_PATH = DATA_ROOT / "common_voice"
AI_PATH = DATA_ROOT / "generated_ai"
TARGET_SR = 16000

# HOW MANY HUMAN FILES TO KEEP PER LANGUAGE?
# We match this roughly to your AI files (200) to keep training balanced.
HUMAN_LIMIT_PER_LANG = 200 

def process_asvspoof():
    print("Processing ASVspoof 2019 LA...")
    protocol_path = ASV_PATH / "ASVspoof2019_LA_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    audio_dir = ASV_PATH / "ASVspoof2019_LA_train/flac"
    
    if not protocol_path.exists():
        print(f"Skipping ASVspoof: Protocol file not found.")
        return []

    data = []
    with open(protocol_path, 'r') as f:
        lines = f.readlines()
        
    # Take a random sample of ASVspoof so it doesn't overwhelm the Indian data
    # (Taking 1000 lines randomly)
    random.shuffle(lines)
    lines = lines[:1000]

    for line in tqdm(lines):
        parts = line.strip().split()
        filename = parts[1]
        label_str = parts[-1]
        label = 0 if label_str == 'bonafide' else 1
        
        file_path = audio_dir / f"{filename}.flac"
        if file_path.exists():
            data.append({"path": str(file_path), "label": label, "source": "asvspoof"})
    return data

def process_common_voice_robust(lang_code):
    print(f"Processing Common Voice ({lang_code})...")
    lang_dir = CV_PATH / lang_code
    
    if not lang_dir.exists():
        print(f"❌ Skipping {lang_code}: Folder not found")
        return []

    # Find ALL audio files recursively (mp3 or wav)
    audio_files = list(lang_dir.rglob("*.mp3")) + list(lang_dir.rglob("*.wav"))
    
    if not audio_files:
        print(f"❌ No audio files found in {lang_dir}")
        return []

    print(f"   Found {len(audio_files)} files. Selecting {HUMAN_LIMIT_PER_LANG}...")
    
    # Shuffle and limit
    random.shuffle(audio_files)
    selected_files = audio_files[:HUMAN_LIMIT_PER_LANG]
    
    data = []
    for file_path in tqdm(selected_files):
        # Convert to WAV 16kHz if needed
        output_path = file_path.with_suffix(".wav")
        
        # If it's an MP3 or needs converting, do it
        if file_path.suffix.lower() == ".mp3" and not output_path.exists():
            try:
                y, sr = librosa.load(file_path, sr=TARGET_SR)
                sf.write(output_path, y, TARGET_SR)
                final_path = output_path
            except Exception:
                continue
        else:
            final_path = file_path

        data.append({
            "path": str(final_path),
            "label": 0,  # HUMAN
            "source": f"cv_{lang_code}"
        })
            
    return data

def process_generated_ai(lang_code):
    print(f"Processing Generated AI ({lang_code})...")
    lang_dir = AI_PATH / lang_code
    
    if not lang_dir.exists():
        return []
    
    data = []
    files = list(lang_dir.glob("*.mp3")) + list(lang_dir.glob("*.wav"))
    
    for file_path in tqdm(files):
        wav_path = file_path.with_suffix('.wav')
        if not wav_path.exists():
            try:
                y, sr = librosa.load(file_path, sr=TARGET_SR)
                sf.write(wav_path, y, TARGET_SR)
            except Exception:
                continue
                
        data.append({
            "path": str(wav_path),
            "label": 1,  # FAKE
            "source": f"gtts_{lang_code}"
        })
    
    return data

if __name__ == "__main__":
    all_data = []
    
    # 1. Base English Data
    all_data.extend(process_asvspoof())
    
    # 2. Indian Languages
    langs = ["hi", "ta", "ml", "te"]
    for lang in langs:
        all_data.extend(process_common_voice_robust(lang)) # Label 0
        all_data.extend(process_generated_ai(lang))        # Label 1

    # 3. Save
    df = pd.DataFrame(all_data)
    if len(df) > 0:
        df.to_csv("train_manifest.csv", index=False)
        print(f"\n✅ SUCCESS! Saved {len(df)} files.")
        print(df['label'].value_counts())
    else:
        print("\n❌ Error: No data found.")