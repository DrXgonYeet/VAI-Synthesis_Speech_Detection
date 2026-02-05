"""
Voice Authentication API - Production Version
Model: VoiceResNet (Fine-Tuned on ASVspoof + Common Voice + AI Generated)
"""

from fastapi import FastAPI, HTTPException, Header, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import base64
import numpy as np
import librosa
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os
import time
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_RATE = 16000
AUDIO_DURATION = 4
TARGET_LENGTH = SAMPLE_RATE * AUDIO_DURATION

# Updated languages list
VALID_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
VALID_API_KEYS = set([os.getenv("API_KEY", "demo-key-123")])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# MODEL ARCHITECTURE (Must match train.py EXACTLY)
# ============================================================================

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

# ============================================================================
# DATA MODELS
# ============================================================================

class VoiceAnalysisRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded audio file")
    language: str = Field(default="English", description="Target Language")

class VoiceAnalysisResponse(BaseModel):
    classification: str
    confidence: float
    processing_time_ms: float
    language: str
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    @staticmethod
    def extract_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        # Same logic as train.py
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize (Important: Match training normalization)
        mean = np.mean(mel_spec_db)
        std = np.std(mel_spec_db)
        mel_spec_db = (mel_spec_db - mean) / (std + 1e-6)
        
        return mel_spec_db

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelManager:
    def __init__(self):
        self.model = None
        self.device = DEVICE
        self.is_loaded = False
    
    def load_model(self):
        try:
            # Initialize the REAL architecture
            self.model = VoiceResNet().to(self.device)
            
            # Load weights
            if os.path.exists("voice_auth_model.pth"):
                self.model.load_state_dict(
                    torch.load("voice_auth_model.pth", map_location=self.device)
                )
                self.model.eval()
                self.is_loaded = True
                logger.info("✅ SUCCESS: Trained VoiceResNet loaded!")
            else:
                logger.error("❌ ERROR: 'voice_auth_model.pth' not found.")
                # We do NOT fall back to dummy weights anymore.
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, mel_spectrogram: np.ndarray) -> tuple:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Add batch and channel dimensions (1, 1, 128, time)
        mel_spec_tensor = torch.FloatTensor(mel_spectrogram).unsqueeze(0).unsqueeze(0)
        mel_spec_tensor = mel_spec_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(mel_spec_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            # Class 0 = Human, Class 1 = Spoof (AI)
            prob_ai = probabilities[0, 1].item()
            prob_human = probabilities[0, 0].item()
        
        return prob_ai, max(prob_ai, prob_human)

# ============================================================================
# AUDIO PROCESSING
# ============================================================================

class AudioProcessor:
    @staticmethod
    def decode_base64_audio(audio_base64: str) -> bytes:
        return base64.b64decode(audio_base64)
    
    @staticmethod
    def load_audio(audio_bytes: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            audio, _ = librosa.load(tmp_path, sr=sr)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        return audio
    
    @staticmethod
    def preprocess_audio(audio: np.ndarray, target_length: int = TARGET_LENGTH) -> np.ndarray:
        # Pad or Truncate to exactly 4 seconds
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        return audio

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(title="Voice Auth API (ResNet)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model_manager = ModelManager()
audio_processor = AudioProcessor()
feature_extractor = FeatureExtractor()

@app.on_event("startup")
async def startup_event():
    model_manager.load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "failed",
        model_loaded=model_manager.is_loaded,
        device=str(DEVICE)
    )

@app.post("/analyze-voice", response_model=VoiceAnalysisResponse)
async def analyze_voice(request: VoiceAnalysisRequest, x_api_key: Optional[str] = Header(None)):
    start_time = time.time()
    
    if not x_api_key or x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # 1. Decode & Load
        audio_bytes = audio_processor.decode_base64_audio(request.audio_base64)
        audio = audio_processor.load_audio(audio_bytes)
        
        # 2. Preprocess (Pad/Crop)
        audio = audio_processor.preprocess_audio(audio)
        
        # 3. Features
        mel_spec = feature_extractor.extract_mel_spectrogram(audio)
        
        # 4. Predict
        prob_ai, confidence = model_manager.predict(mel_spec)
        
        # 5. Result
        classification = "AI_GENERATED" if prob_ai > 0.5 else "HUMAN"
        
        processing_time = (time.time() - start_time) * 1000
        
        return VoiceAnalysisResponse(
            classification=classification,
            confidence=round(confidence, 4),
            processing_time_ms=round(processing_time, 2),
            language=request.language,
            model_version="VoiceResNet-FineTuned-v1"
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)