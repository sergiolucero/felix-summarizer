import streamlit as st
import speech_recognition as sr
from transformers import pipeline

summarizer = pipeline("summarization")
recognizer = sr.Recognizer()

def audio_to_text(audio_file):
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

st.title("Audio Summarizer")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    with st.spinner("Converting audio to text..."):
        extracted_text = audio_to_text(uploaded_file)
    st.write("Extracted Text:")
    st.write(extracted_text)
    
    with st.spinner("Generating summary..."):
        summary = summarizer(extracted_text, max_length=100, min_length=30, do_sample=False)
    st.write("Summary:")
    st.write(summary[0]['summary_text'])

