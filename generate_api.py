# generate_ai.py
from gtts import gTTS
import os
from tqdm import tqdm
import time

# CONFIGURATION
OUTPUT_DIR = "data/generated_ai"
SAMPLES_PER_LANG = 200  # Start with 50 samples per language (adjust as needed)

# Sample text data (generic phrases)
TEXT_DATA = {
    "hi": [ # Hindi
        "नमस्ते, आप कैसे हैं?",
        "यह एक परीक्षण संदेश है।",
        "मौसम आज बहुत अच्छा है।",
        "कृपया दरवाजा बंद करें।",
        "मैं मशीन लर्निंग सीख रहा हूं।",
        "भारत एक विशाल देश है।",
        "मुझे संगीत सुनना पसंद है।",
        "क्या आप मेरी मदद कर सकते हैं?",
        "यह आर्टिफिशियल इंटेलिजेंस की आवाज़ है।",
        "सुप्रभात, आपका दिन शुभ हो।"
    ],
    "ta": [ # Tamil
        "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "இது ஒரு சோதனை செய்தி.",
        "இன்று வானிலை மிகவும் நன்றாக இருக்கிறது.",
        "தயவுசெய்து கதவை மூடவும்.",
        "நான் இயந்திர கற்றலைக் கற்றுக்கொள்கிறேன்.",
        "இந்தியா ஒரு பெரிய நாடு.",
        "எனக்கு இசை கேட்க பிடிக்கும்.",
        "நீங்கள் எனக்கு உதவ முடியுமா?",
        "இது செயற்கை நுண்ணறிவின் குரல்.",
        "காலை வணக்கம், இனிய நாள்."
    ],
    "ml": [ # Malayalam
        "നമസ്കാരം, സുഖമാണോ?",
        "ഇതൊരു പരീക്ഷണ സന്ദേശമാണ്.",
        "ഇന്ന് നല്ല കാലാവസ്ഥയാണ്.",
        "ദയവായി വാതിൽ അടയ്ക്കൂ.",
        "ഞാൻ മെഷീൻ ലേണിംഗ് പഠിക്കുകയാണ്.",
        "ഇന്ത്യ ഒരു വലിയ രാജ്യമാണ്.",
        "എനിക്ക് പാട്ട് കേൾക്കാൻ ഇഷ്ടമാണ്.",
        "നിങ്ങൾക്ക് എന്നെ സഹായിക്കാമോ?",
        "ഇതൊരു ആർട്ടിഫിഷ്യൽ ഇന്റലിജൻസ് ശബ്ദമാണ്.",
        "സുപ്രഭാതം, നല്ലൊരു ദിവസം ആശംസിക്കുന്നു."
    ],
    "te": [ # Telugu
        "నమస్కారం, మీరు ఎలా ఉన్నారు?",
        "ఇది ఒక పరీక్ష సందేశం.",
        "ఈ రోజు వాతావరణం చాలా బాగుంది.",
        "దయచేసి తలుపు మూయండి.",
        "నేను మెషిన్ లెర్నింగ్ నేర్చుకుంటున్నాను.",
        "భారతదేశం చాలా పెద్ద దేశం.",
        "నాకు సంగీతం వినడం ఇష్టం.",
        "మీరు నాకు సహాయం చేయగలరా?",
        "ఇది కృత్రిమ మేధస్సు యొక్క స్వరం.",
        "శుభోదయం, మీ రోజు బాగుండాలి."
    ]
}

def generate_clips():
    print("🚀 Starting AI Voice Generation...")
    
    for lang_code, sentences in TEXT_DATA.items():
        print(f"\nProcessing Language: {lang_code}")
        
        # Create folder: data/generated_ai/hi, etc.
        lang_dir = os.path.join(OUTPUT_DIR, lang_code)
        os.makedirs(lang_dir, exist_ok=True)
        
        count = 0
        # Loop to generate enough samples
        while count < SAMPLES_PER_LANG:
            for i, text in enumerate(sentences):
                if count >= SAMPLES_PER_LANG:
                    break
                
                filename = f"ai_{lang_code}_{count}.mp3"
                filepath = os.path.join(lang_dir, filename)
                
                # Skip if already exists
                if os.path.exists(filepath):
                    count += 1
                    continue
                
                try:
                    # Generate Audio
                    tts = gTTS(text=text, lang=lang_code, slow=False)
                    tts.save(filepath)
                    count += 1
                    print(f"[{count}/{SAMPLES_PER_LANG}] Saved: {filename}", end='\r')
                    
                    # Sleep briefly to avoid Google API banning us
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"\nError creating {filename}: {e}")
                    time.sleep(2) # Wait longer if error occurs

    print("\n\n✅ Generation Complete! Check the 'data/generated_ai' folder.")

if __name__ == "__main__":
    generate_clips()