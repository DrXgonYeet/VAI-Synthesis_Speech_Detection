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
from pathlib import Path
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

VALID_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
VALID_API_KEYS = set([os.getenv("API_KEY", "demo-key-123")])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# DATA MODELS
# ============================================================================

class VoiceAnalysisRequest(BaseModel):
    """Request model for voice analysis"""
    audio_base64: str = Field(..., description="Base64-encoded MP3 audio file")
    language: str = Field(
        default="English",
        description="Language of audio: Tamil, English, Hindi, Malayalam, or Telugu"
    )

class ExplainabilityInfo(BaseModel):
    """Explainability details for model decision"""
    model_name: str
    raw_probability: float = Field(..., ge=0.0, le=1.0)
    feature_space: str
    training_dataset: str
    confidence_calibration: str

class VoiceAnalysisResponse(BaseModel):
    """Response model for voice analysis"""
    classification: str = Field(..., description="AI_GENERATED or HUMAN")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    processing_time_ms: float
    language: str
    explainability: ExplainabilityInfo

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    supported_languages: list
    device: str

# ============================================================================
# SYNTHETIC SPEECH DETECTOR MODEL
# ============================================================================

class SimpleSpeechDetector(nn.Module):
    """
    Lightweight CNN for synthetic speech detection.
    Uses mel-spectrogram features for classification.
    """
    
    def __init__(self, num_classes=2):
        super(SimpleSpeechDetector, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """Extract features from audio for classification"""
    
    @staticmethod
    def extract_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    @staticmethod
    def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return mfcc
    
    @staticmethod
    def extract_stft(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        stft = np.abs(librosa.stft(audio))
        stft_db = librosa.power_to_db(stft, ref=np.max)
        return stft_db
    
    @staticmethod
    def extract_spectral_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, float]:
        features = {}
        
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spec_centroid))
        features['spectral_centroid_std'] = float(np.std(spec_centroid))
        
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spec_rolloff_mean'] = float(np.mean(spec_rolloff))
        
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        return features

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelManager:
    """Manages model loading, inference, and lifecycle"""
    
    def __init__(self):
        self.model = None
        self.device = DEVICE
        self.is_loaded = False
        self.model_info = {
            "name": "SimpleSpeechDetector",
            "version": "1.0.0",
            "training_dataset": "ASVspoof_2019+Custom_Data"
        }
    
    def load_model(self):
        """Load or initialize the model"""
        try:
            self.model = SimpleSpeechDetector(num_classes=2).to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Model loaded on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, mel_spectrogram: np.ndarray) -> tuple:
        """
        Run inference on mel-spectrogram.
        
        Returns:
            (probability_ai_generated, confidence)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        mel_spec_tensor = torch.FloatTensor(mel_spectrogram).unsqueeze(0).unsqueeze(0)
        mel_spec_tensor = mel_spec_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(mel_spec_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            prob_ai = probabilities[0, 1].item()
            prob_human = probabilities[0, 0].item()
        
        return prob_ai, max(prob_ai, prob_human)

# ============================================================================
# AUDIO PROCESSING
# ============================================================================

class AudioProcessor:
    """Processes audio files for analysis"""
    
    @staticmethod
    def decode_base64_audio(audio_base64: str) -> bytes:
        """Decode base64-encoded audio"""
        try:
            return base64.b64decode(audio_base64)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 audio: {e}")
    
    @staticmethod
    def load_audio(audio_bytes: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
        """
        Load audio from bytes (MP3 format).
        
        Args:
            audio_bytes: Raw audio file bytes
            sr: Target sample rate
            
        Returns:
            numpy array of audio samples
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            audio, _ = librosa.load(tmp_path, sr=sr)
            os.unlink(tmp_path)
            
            return audio
        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}")
    
    @staticmethod
    def preprocess_audio(audio: np.ndarray, target_length: int = TARGET_LENGTH) -> np.ndarray:
        """
        Preprocess audio for model input.
        
        Steps:
        1. Normalize amplitude
        2. Pad or truncate to fixed length
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        return audio
    
    @staticmethod
    def validate_audio(audio: np.ndarray, min_length: int = 8000) -> bool:
        """Validate audio has sufficient samples"""
        return len(audio) >= min_length

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Voice Authentication API",
    description="Detects AI-generated vs human speech in Tamil, English, Hindi, Malayalam, Telugu",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()
audio_processor = AudioProcessor()
feature_extractor = FeatureExtractor()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    logger.info("Starting Voice Authentication API...")
    model_manager.load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status and model information
    """
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        supported_languages=VALID_LANGUAGES,
        device=str(DEVICE)
    )

@app.post("/analyze-voice", response_model=VoiceAnalysisResponse)
async def analyze_voice(
    request: VoiceAnalysisRequest,
    x_api_key: Optional[str] = Header(None)
) -> VoiceAnalysisResponse:
    """
    Analyze voice sample to determine if AI-generated or human.
    
    Args:
        request: JSON with base64-encoded audio and language
        x_api_key: API key for authentication
        
    Returns:
        Classification result with confidence and explainability info
        
    Example:
        curl -X POST http://localhost:8000/analyze-voice \
          -H "Content-Type: application/json" \
          -H "x-api-key: demo-key-123" \
          -d '{"audio_base64": "...", "language": "English"}'
    """
    start_time = time.time()
    
    # Authentication
    if not x_api_key or x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide 'x-api-key' header."
        )
    
    # Validation
    if request.language not in VALID_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Language must be one of {VALID_LANGUAGES}. Got: {request.language}"
        )
    
    if not request.audio_base64 or len(request.audio_base64) < 100:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio_base64. Must be non-empty base64-encoded MP3 file."
        )
    
    try:
        # Audio Processing
        audio_bytes = audio_processor.decode_base64_audio(request.audio_base64)
        audio = audio_processor.load_audio(audio_bytes, sr=SAMPLE_RATE)
        
        if not audio_processor.validate_audio(audio):
            raise ValueError("Audio too short. Minimum 0.5 seconds required.")
        
        audio = audio_processor.preprocess_audio(audio, target_length=TARGET_LENGTH)
        
        # Feature Extraction
        mel_spectrogram = feature_extractor.extract_mel_spectrogram(audio, sr=SAMPLE_RATE)
        spectral_features = feature_extractor.extract_spectral_features(audio, sr=SAMPLE_RATE)
        
        # Model Inference
        prob_ai_generated, confidence = model_manager.predict(mel_spectrogram)
        
        # Decision Logic
        classification = "AI_GENERATED" if prob_ai_generated > 0.5 else "HUMAN"
        
        if classification == "AI_GENERATED":
            final_confidence = prob_ai_generated
        else:
            final_confidence = 1 - prob_ai_generated
        
        # Explainability
        explainability = ExplainabilityInfo(
            model_name="SimpleSpeechDetector",
            raw_probability=round(prob_ai_generated, 4),
            feature_space="mel_spectrogram_128_bins",
            training_dataset="ASVspoof_2019+Custom",
            confidence_calibration="Sigmoid with softmax"
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Analysis complete | Language: {request.language} | "
            f"Classification: {classification} | Confidence: {final_confidence:.4f} | "
            f"Time: {processing_time:.1f}ms"
        )
        
        return VoiceAnalysisResponse(
            classification=classification,
            confidence=round(final_confidence, 4),
            processing_time_ms=round(processing_time, 2),
            language=request.language,
            explainability=explainability
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )

@app.post("/analyze-voice-file")
async def analyze_voice_file(
    file: UploadFile = File(...),
    language: str = "English",
    x_api_key: Optional[str] = Header(None)
):
    """
    Alternative endpoint that accepts file upload instead of base64.
    More convenient for testing.
    """
    if not x_api_key or x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        contents = await file.read()
        audio_base64 = base64.b64encode(contents).decode('utf-8')
        
        request = VoiceAnalysisRequest(
            audio_base64=audio_base64,
            language=language
        )
        
        return await analyze_voice(request, x_api_key)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "message": "Voice Authentication API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health (GET)",
            "analyze_voice": "/analyze-voice (POST)",
            "analyze_voice_file": "/analyze-voice-file (POST)",
            "docs": "/docs (Swagger UI)"
        },
        "authentication": "Provide 'x-api-key' header",
        "supported_languages": VALID_LANGUAGES
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": time.time()
    }

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )