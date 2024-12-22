import streamlit as st
from datetime import datetime
import json
import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.agents.audio_agent import AudioAgent

load_dotenv()

# Constants
openai_api_key = os.getenv("OPENAI_API_KEY")
output_folder = "./saved_outputs"

# Configure Streamlit app
st.set_page_config(
    page_title="Therapy Session Assistant",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# App title and header
st.title("🌊 Therapy Session Assistant")

# Sidebar for patient information
st.sidebar.header("Patient Information")
patient_name = st.sidebar.text_input("Patient Name", value="John Doe")
patient_age = st.sidebar.number_input("Age", value=32, step=1)
patient_history = st.sidebar.text_area("History", value="Anxiety and mild depression.")

# Initialize AudioAgent
audio_agent = AudioAgent(
    openai_api_key=openai_api_key,
    patient_data={
        "name": patient_name,
        "age": patient_age,
        "history": patient_history,
    },
    target_language="en"
)

# Record Audio
st.header("🎙️ Step 1: Record Audio")
audio_file = None
stop_event = None
recording_thread = None
stream = None
audio = None
frames = []

# Buttons for recording
start_recording = st.button("Start Recording")
stop_recording = st.button("Stop Recording")

# Handle recording
if start_recording:
    with st.spinner("Recording in progress..."):
        stop_event, recording_thread, audio_file, stream, audio, frames = audio_agent.capture_audio()
        st.success("Recording started.")

if stop_recording and audio_file:
    with st.spinner("Stopping recording..."):
        audio_file = audio_agent.stop_recording(stop_event, recording_thread, stream, audio, audio_file, frames)
        st.success(f"Recording saved: {audio_file}")

# Step 2: Transcription and Summarization
if audio_file:
    st.header("📝 Step 2: Transcription and Summarization")
    with st.spinner("Processing audio..."):
        transcription_result = audio_agent.transcribe_audio(audio_file)

    if transcription_result:
        st.subheader("Transcription")
        st.text_area("Transcript", transcription_result, height=200)

        with st.spinner("Generating session summary..."):
            summary = audio_agent.summarize_transcript(transcription_result)

        if summary:
            st.subheader("Session Summary")
            st.markdown(f"**Date**: {datetime.today().strftime('%Y-%m-%d')}")
            st.markdown("**Overview**")
            st.write(summary)
            
            # Export or download summary
            session_date = datetime.today().strftime('%Y-%m-%d')
            file_name = f"{patient_name.replace(' ', '_')}_session_summary_{session_date}.json"
            json_data = json.dumps(summary, indent=4)

            st.download_button(
                label="Download Summary as JSON",
                data=json_data,
                file_name=file_name,
                mime="application/json",
            )
        else:
            st.error("Summary generation failed.")
    else:
        st.error("Transcription failed. Please try again.")
