import openai
import pyaudio
import wave
import datetime
import os
import logging
from dotenv import load_dotenv
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "../saved_outputs/")  # Default path if not provided

class AudioAgent:
    def __init__(self, openai_api_key, patient_data, target_language="en"):
        """
        Initialize the AudioAgent with OpenAI API credentials and patient data.

        Args:
            openai_api_key (str): OpenAI API key.
            patient_data (dict): Patient-specific information for contextual summaries.
            target_language (str): Language to translate the transcription into (default is 'en' for English).
        """
        self.openai_api_key = openai_api_key
        self.patient_data = patient_data
        self.target_language = target_language
        openai.api_key = self.openai_api_key

    def capture_audio(self, chunk_size=1024, rate=44100, recording_duration=30):
        """
        Captures real-time audio from the microphone and saves it to a file.
        The recording will stop after a specified duration.

        Returns:
            str: Path to the saved audio file.
        """
        audio_format = pyaudio.paInt16
        channels = 1
        
        # Define dynamic output folder and file name
        output_folder = os.path.join(OUTPUT_PATH, 'audio_records', self.patient_data["name"])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        output_file = os.path.join(output_folder, f'{self.patient_data["name"]}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.wav')

        audio = pyaudio.PyAudio()
        stream = audio.open(format=audio_format, channels=channels, 
                            rate=rate, input=True, frames_per_buffer=chunk_size)

        frames = []

        st.write(f"Recording for {recording_duration} seconds...")

        # Record for the specified duration
        for _ in range(0, int(rate / chunk_size * recording_duration)):
            data = stream.read(chunk_size)
            frames.append(data)

        # Clean up and save the audio
        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(audio_format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        logging.info(f"Recording saved to {output_file}")
        return output_file

    def transcribe_audio(self, audio_file):
        """
        Transcribes audio to text using OpenAI's Whisper API.

        Returns:
            str: Transcribed text.
        """
        try:
            logging.info(f"Transcribing {audio_file}...")
            with open(audio_file, "rb") as file:
                response = openai.Audio.transcribe(model="whisper-1", file=file)
            return response.get("text", "")
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return None

    def summarize_transcript(self, transcript):
        """
        Summarizes the transcribed speech using OpenAI's GPT API, incorporating patient data.

        Args:
        transcript (str): Transcribed speech text.

        Returns:
        dict: A summary of the session in structured JSON format.
        """
        patient_context = f"Patient Name: {self.patient_data.get('name', 'Unknown')}, " \
                          f"Age: {self.patient_data.get('age', 'Unknown')}, " \
                          f"History: {self.patient_data.get('history', 'No history provided')}"

        preamble = (
            "You are an assistant helping a therapist interpret patient speech. Your task is to summarize a therapy session transcript "
            "into key insights and actionable points that reflect the patient's emotional state, challenges, progress, and any therapeutic "
            "goals mentioned. The summary should be structured and follow the format provided below. Here are some examples of valid summaries:\n\n"
            "Example 1:\n"
            "Overview: The patient discussed their recent trauma-related anxiety and fears of relapse. They mentioned feeling overwhelmed and not "
            "able to manage their emotions during stressful situations.\n"
            "Key Insights: The patient needs further support in developing coping strategies for anxiety, specifically in high-stress environments.\n"
            "Emotions or States: Anxiety, frustration, and hopelessness were identified.\n"
            "Therapeutic Goals: To work on relaxation techniques, mindfulness, and building emotional resilience.\n\n"
            "Example 2:\n"
            "Overview: The patient shared their experience of improving their OCD symptoms but still struggles with intrusive thoughts and compulsive behaviors.\n"
            "Key Insights: The patient showed progress in managing their symptoms, but they need more consistent practice with cognitive-behavioral techniques.\n"
            "Emotions or States: Frustration with lack of full control over compulsions.\n"
            "Therapeutic Goals: To continue implementing exposure therapy and work on resisting compulsive rituals.\n\n"
            "Now, please summarize the following therapy session transcript into key insights and actionable points. Focus on the patient's emotional state, "
            "challenges, progress, and any therapeutic goals mentioned.\n\n"
            f"Patient Context:\n{patient_context}\n\n"
            f"Transcript:\n{transcript}\n\n"
            f"Provide the summary in the following structure:\n"
            f"1. **Overview**: Main themes or issues discussed.\n"
            f"2. **Key Insights**: Actionable takeaways.\n"
            f"3. **Emotions or States**: Emotional states identified.\n"
            f"4. **Therapeutic Goals**: Goals or actions for future sessions.\n"
        )

        try:
            logging.info("Generating summary...")
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": preamble}]
            )
            summary = response.choices[0].message['content'] if response else ""
            return {
                "patient_name": self.patient_data.get("name"),
                "date": str(datetime.date.today()),
                "summary": summary
            }
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return None

# Streamlit interface
def main():
    st.title("Therapy Session Recording, Transcription, and Summary")

    # Sample patient data
    patient_data = {
        "name": st.text_input("Enter patient name", "John Doe"),
        "age": st.number_input("Enter patient age", min_value=1, max_value=120, value=30),
        "history": st.text_area("Enter patient history", "No significant medical history.")
    }

    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")

    if openai_api_key:
        audio_agent = AudioAgent(openai_api_key=openai_api_key, patient_data=patient_data)

        if st.button("Start Recording"):
            # Capture audio for a fixed duration
            audio_file = audio_agent.capture_audio(recording_duration=30)
            st.write(f"Audio saved to: {audio_file}")

            # Transcribe the audio
            transcription = audio_agent.transcribe_audio(audio_file)
            st.subheader("Transcription")
            st.text_area("Transcribed Text", transcription, height=300)

            # Generate session summary
            summary = audio_agent.summarize_transcript(transcription)
            st.subheader("Session Summary")
            st.write(summary['summary'])

            st.success(f"Summary generated successfully!")

if __name__ == "__main__":
    main()
