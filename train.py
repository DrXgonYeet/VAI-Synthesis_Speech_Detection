# train.py (FOR USER 2 ONLY - FINE TUNING VERSION)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import os

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001  # <--- CRITICAL: Lower learning rate for fine-tuning
SAMPLE_RATE = 16000
DURATION = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# MODEL ARCHITECTURE (Must match User 1 exactly)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class VoiceResNet(nn.Module):
    def __init__(self):
        super(VoiceResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(32, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 2)
    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# ==========================================
# DATASET CLASS
# ==========================================
class AudioDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.target_len = SAMPLE_RATE * DURATION
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = row['path']
        label = row['label']
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
            if len(audio) < self.target_len:
                audio = np.pad(audio, (0, self.target_len - len(audio)))
            else:
                audio = audio[:self.target_len]
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mean = np.mean(mel_spec_db)
            std = np.std(mel_spec_db)
            mel_spec_db = (mel_spec_db - mean) / (std + 1e-6)
            spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)
            return spec_tensor, torch.tensor(label, dtype=torch.long)
        except Exception:
            return torch.zeros((1, 128, 126)), torch.tensor(0, dtype=torch.long)

# ==========================================
# TRAINING LOOP
# ==========================================
def train():
    print(f"Initializing Fine-Tuning on {DEVICE}...")
    
    if not os.path.exists("train_manifest.csv"):
        print("Error: train_manifest.csv not found. Run data_prep.py first.")
        return

    dataset = AudioDataset("train_manifest.csv")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    model = VoiceResNet().to(DEVICE)
    
    # ---------------------------------------------------------
    # LOAD USER 1's MODEL
    # ---------------------------------------------------------
    pretrained_weights = "voice_auth_model.pth"
    if os.path.exists(pretrained_weights):
        print(f"✅ Found base model: {pretrained_weights}")
        try:
            model.load_state_dict(torch.load(pretrained_weights, map_location=DEVICE))
            print("✅ Weights loaded! The model now knows English/ASVspoof.")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return
    else:
        print("❌ STOP! 'voice_auth_model.pth' is missing.")
        print("You must wait for User 1 to send you this file.")
        return
    # ---------------------------------------------------------

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    print(f"Starting Fine-Tuning for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1} Results -> Val Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "voice_auth_model_final.pth")
            print("✅ Improved Model Saved as 'voice_auth_model_final.pth'")

    print("\nTraining Complete! Use 'voice_auth_model_final.pth' for your API.")

if __name__ == "__main__":
    train()