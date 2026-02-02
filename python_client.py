import requests
import base64
import os
import logging
from typing import Dict, Any

# -----------------------------------------------------------------------------
# LOGGING CONFIG
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("VoiceAuthClient")

# -----------------------------------------------------------------------------
# VOICE AUTH API CLIENT
# -----------------------------------------------------------------------------
class VoiceAuthAPI:
    """
    Client for Voice Authentication API.
    This client WILL NOT change when the model is replaced.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "demo-key-123",
        timeout: int = 10
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }

    # -------------------------------------------------------------------------
    # HEALTH CHECK
    # -------------------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health and model status.
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=5
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"Health check failed: {e}")
            raise RuntimeError("API is not reachable") from e

    # -------------------------------------------------------------------------
    # AUDIO ANALYSIS
    # -------------------------------------------------------------------------
    def analyze_audio(
        self,
        audio_file_path: str,
        language: str = "English"
    ) -> Dict[str, Any]:
        """
        Analyze an audio file and return full API response.
        """

        # Validate file
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        # Read and encode audio
        try:
            with open(audio_file_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            raise RuntimeError("Failed to read or encode audio file") from e

        payload = {
            "audio_base64": audio_base64,
            "language": language
        }

        try:
            response = requests.post(
                f"{self.base_url}/analyze-voice",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"API Error {response.status_code}: {response.text}"
                )

            return response.json()

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise RuntimeError("Failed to connect to API") from e

    # -------------------------------------------------------------------------
    # SIMPLE DECISION API
    # -------------------------------------------------------------------------
    def is_human_voice(
        self,
        audio_file_path: str,
        language: str = "English",
        policy: str = "normal"
    ) -> bool:
        """
        High-level helper function.
        Returns True if voice is classified as HUMAN with sufficient confidence.
        """

        thresholds = {
            "strict": 0.85,
            "normal": 0.70,
            "lenient": 0.55
        }

        threshold = thresholds.get(policy, 0.70)

        try:
            result = self.analyze_audio(audio_file_path, language)

            classification = result.get("classification")
            confidence = result.get("confidence", 0.0)

            logger.info(
                f"Decision | classification={classification}, "
                f"confidence={confidence:.3f}, threshold={threshold}"
            )

            return (
                classification == "HUMAN" and
                confidence >= threshold
            )

        except Exception as e:
            logger.error(f"Voice verification failed: {e}")
            return False

# -----------------------------------------------------------------------------
# USAGE EXAMPLE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    api = VoiceAuthAPI()

    print("\n--- HEALTH CHECK ---")
    try:
        health = api.health_check()
        print(health)
    except Exception as e:
        print("API not available:", e)
        exit(1)

    print("\n--- ANALYZE AUDIO ---")
    try:
        result = api.analyze_audio("sample.mp3", "Tamil")
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Processing Time: {result['processing_time_ms']} ms")
    except Exception as e:
        print("Analysis failed:", e)

    print("\n--- SIMPLE HUMAN CHECK ---")
    is_human = api.is_human_voice("sample.mp3", policy="normal")
    print(f"Is human voice? {is_human}")
