import streamlit as st
from datetime import datetime
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.agents.audio_agent import AudioAgent

# Load environment variables
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

# Load custom CSS
with open("styles/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

# Step 1: Record Audio
st.header("🎙️ Step 1: Record Audio")
audio_file = None

# Start recording
start_recording = st.button("Start Recording")
stop_recording = st.button("Stop Recording")

# Handle recording and stopping
if start_recording:
    with st.spinner("Recording in progress..."):
        audio_file = audio_agent.capture_audio(duration=60)  # Start recording
    st.success("Recording complete! Audio saved as `recorded_audio.wav`.")

elif stop_recording and audio_file is None:
    with st.spinner("Stopping recording..."):
        # Stop the recording immediately and process the audio
        audio_file = audio_agent.capture_audio(duration=0)  # Stop recording immediately
    st.success("Recording stopped and saved.")

# Step 2: Transcription and Translation
if audio_file:
    st.header("📝 Step 2: Transcription and Translation")
    transcription_result = audio_agent.transcribe_audio(audio_file)

    if transcription_result:
        st.subheader("Transcription")
        st.text_area("Transcript", transcription_result, height=200)

        translated_text = audio_agent.translate_text(transcription_result)
        st.subheader("Translated Text")
        st.text_area("Translated", translated_text, height=200)
    else:
        st.error("Transcription failed. Please try again.")

# Step 3: Generate Session Summary
if audio_file and transcription_result:
    st.header("📋 Step 3: Generate Session Summary")
    
    # Call AudioAgent to generate session summary after analyzing the transcription
    summary = audio_agent.summarize_transcript(transcription_result)

    if summary:
        st.subheader("Session Summary")
        st.markdown(f"**Date**: {datetime.today().strftime('%Y-%m-%d')}")
        st.markdown("**Overview**")
        st.write(summary.get("overview", "N/A"))
        st.markdown("**Key Insights**")
        st.write(summary.get("key_insights", "N/A"))
        st.markdown("**Emotions or States**")
        st.write(summary.get("emotions_or_states", "N/A"))
        st.markdown("**Therapeutic Goals**")
        st.write(summary.get("therapeutic_goals", "N/A"))

        # Generate a unique file name for the session summary based on patient name and date
        session_date = datetime.today().strftime('%Y-%m-%d')
        file_name = f"{patient_name.replace(' ', '_')}_session_summary_{session_date}.json"

        # Ensure the "fiches" directory exists
        fiches_folder = './fiches'
        if not os.path.exists(fiches_folder):
            os.makedirs(fiches_folder)

        # Save the summary to the "fiches" directory
        if st.button("Save Summary"):
            audio_agent.save_summary_to_json(summary, output_folder=fiches_folder, file_name=file_name)
            st.success(f"Summary saved to `{os.path.join(fiches_folder, file_name)}`.")

        # Display full session information with all analyzed fields from the second LLM
        st.header("🎯 Full Session Representation")
        session_data = {
            "Patient Name": patient_name,
            "Age": patient_age,
            "History": patient_history,
            "Date": session_date,
            "Overview": summary.get("overview", "N/A"),
            "Key Insights": summary.get("key_insights", "N/A"),
            "Emotions or States": summary.get("emotions_or_states", "N/A"),
            "Therapeutic Goals": summary.get("therapeutic_goals", "N/A")
        }
        
        # Show the full session information
        st.json(session_data)

    else:
        st.error("Summary generation failed.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ by your Therapy Assistant.")
