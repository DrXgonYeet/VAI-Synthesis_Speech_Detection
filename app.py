import streamlit as st
import base64
import numpy as np
import librosa
from main import AudioProcessor, FeatureExtractor, model_manager
import matplotlib.pyplot as plt

# Initialize Model (Must load once)
if not model_manager.is_loaded:
    model_manager.load_model()

st.set_page_config(page_title="VAI Speech Detector", page_icon="🎙️")

st.title("🎙️ VAI Speech Detection")
st.write("Upload an audio file to determine if it is Human or AI-Generated.")

uploaded_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # 1. Prepare Audio for your existing logic
    audio_bytes = uploaded_file.getvalue()
    
    # Use your existing AudioProcessor logic
    audio = AudioProcessor.load_audio(audio_bytes)
    audio = AudioProcessor.preprocess_audio(audio)
    
    st.audio(audio_bytes)
    if st.button("Run Detection"):
        with st.spinner("Analyzing audio..."):
            try:
                # 2. Extract Features using your existing code
                mel_spec = FeatureExtractor.extract_mel_spectrogram(audio)
                fig, ax = plt.subplots()
                img = librosa.display.specshow(mel_spec, ax=ax)
                st.pyplot(fig)
                # 3. Predict using your existing model_manager
                prob_ai, confidence = model_manager.predict(mel_spec)
                
                # 4. Display Results
                classification = "AI_GENERATED" if prob_ai > 0.5 else "HUMAN"
                
                if classification == "AI_GENERATED":
                    st.error(f"⚠️ {classification} (Confidence: {confidence:.2%})")
                else:
                    st.success(f"✅ {classification} (Confidence: {confidence:.2%})")
                    
            except Exception as e:
                st.error(f"Error processing audio: {e}")