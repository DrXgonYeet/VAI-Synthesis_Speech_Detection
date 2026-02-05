import requests
import base64

# Your LIVE URL
API_URL = "https://vai-synthesis-speech-detection.onrender.com/analyze-voice"
API_KEY = "API_KEY"  # The key you set in Render

# Path to a test file (ensure this file exists!)
TEST_FILE = "test_human.wav" # OR "popli.mp3"

def test_api():
    print(f"🚀 Connecting to: {API_URL} ...")
    
    # 1. Read Audio
    try:
        with open(TEST_FILE, "rb") as f:
            audio_data = f.read()
        b64_audio = base64.b64encode(audio_data).decode("utf-8")
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{TEST_FILE}'. Please check the filename.")
        return

    # 2. Send Request
    payload = {
        "audio_base64": b64_audio,
        "language": "English"
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        
        print(f"📡 Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✅ SUCCESS! Response:")
            print(response.json())
        else:
            print("❌ FAILED. Response:")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_api()