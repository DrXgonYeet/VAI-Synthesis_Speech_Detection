import requests
import base64
import json

class VoiceAuthAPI:
    def __init__(self, base_url="http://localhost:8000", api_key="demo-key-123"):
        self.base_url = base_url
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }
    
    def health_check(self):
        """Check API status"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def analyze_audio(self, audio_file_path, language="English"):
        """Analyze audio file"""
        # Read and encode audio file
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        payload = {
            "audio_base64": audio_base64,
            "language": language
        }
        
        response = requests.post(
            f"{self.base_url}/analyze-voice",
            json=payload,
            headers=self.headers
        )
        
        return response.json()
    
    def is_human_voice(self, audio_file_path, language="English", confidence_threshold=0.7):
        """Simplified check: returns True if human voice"""
        result = self.analyze_audio(audio_file_path, language)
        return (
            result["classification"] == "HUMAN" and 
            result["confidence"] >= confidence_threshold
        )

# Usage
api = VoiceAuthAPI()

# Check health
print(api.health_check())

# Analyze audio
result = api.analyze_audio("sample.mp3", "Tamil")
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.2%}")

# Simple check
is_human = api.is_human_voice("sample.mp3")
print(f"Is human voice? {is_human}")
