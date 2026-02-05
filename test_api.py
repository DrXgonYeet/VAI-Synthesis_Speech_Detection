"""
Test script for Voice Authentication API
Run: python test_api.py
"""

import requests
import base64
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import tempfile

API_URL = "https://vai-synthesis-speech-detection.onrender.com"
API_KEY = "0713hex_luthor"

HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

def test_health_check():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def create_test_audio(duration: float = 2.0, freq: float = 440) -> str:
    """Create a simple test audio (sine wave) and return as base64"""
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    audio += 0.1 * np.sin(2 * np.pi * freq * 2 * t)
    audio += 0.05 * np.sin(2 * np.pi * freq * 3 * t)
    
    audio = np.int16(audio * 32767)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wavfile.write(tmp.name, sr, audio)
        
        with open(tmp.name, 'rb') as f:
            audio_bytes = f.read()
        
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        return base64_audio

def test_analyze_voice(language: str = "English"):
    """Test voice analysis endpoint"""
    print("\n" + "="*60)
    print(f"TEST: Analyze Voice ({language})")
    print("="*60)
    
    try:
        audio_base64 = create_test_audio(duration=2.0, freq=440)
        
        payload = {
            "audio_base64": audio_base64,
            "language": language
        }
        
        response = requests.post(
            f"{API_URL}/analyze-voice",
            json=payload,
            headers=HEADERS
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nResponse:")
            print(json.dumps(result, indent=2))
            
            assert "classification" in result, "Missing 'classification' field"
            assert "confidence" in result, "Missing 'confidence' field"
            assert result["classification"] in ["AI_GENERATED", "HUMAN"]
            assert 0.0 <= result["confidence"] <= 1.0
            
            print("\n✅ Response structure valid")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_all_languages():
    """Test all supported languages"""
    print("\n" + "="*60)
    print("TEST: All Supported Languages")
    print("="*60)
    
    languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    results = {}
    
    for language in languages:
        print(f"\nTesting {language}...", end=" ")
        audio_base64 = create_test_audio(duration=1.5)
        
        payload = {
            "audio_base64": audio_base64,
            "language": language
        }
        
        response = requests.post(
            f"{API_URL}/analyze-voice",
            json=payload,
            headers=HEADERS
        )
        
        success = response.status_code == 200
        results[language] = success
        print(f"{'✅' if success else '❌'}")
    
    print(f"\nResults: {sum(results.values())}/{len(results)} passed")
    return all(results.values())

def test_invalid_api_key():
    """Test with invalid API key"""
    print("\n" + "="*60)
    print("TEST: Invalid API Key")
    print("="*60)
    
    audio_base64 = create_test_audio(duration=1.0)
    
    payload = {
        "audio_base64": audio_base64,
        "language": "English"
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "invalid-key-12345"
    }
    
    response = requests.post(
        f"{API_URL}/analyze-voice",
        json=payload,
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    success = response.status_code == 401
    print(f"{'✅' if success else '❌'} Correctly rejected invalid API key")
    return success

def run_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("VOICE AUTHENTICATION API - TEST SUITE")
    print("="*60)
    print(f"API URL: {API_URL}")
    print(f"API Key: {API_KEY}")
    
    tests = [
        ("Health Check", test_health_check),
        ("Analyze Voice - English", lambda: test_analyze_voice("English")),
        ("Analyze Voice - Tamil", lambda: test_analyze_voice("Tamil")),
        ("Invalid API Key", test_invalid_api_key),
        ("All Languages", test_all_languages),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Cannot connect to {API_URL}")
            print("Make sure the API is running: python main.py")
            return
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            results[test_name] = False
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

if __name__ == "__main__":
    run_tests()
