import openai
import json
import pyaudio
import wave
import datetime
import os
import logging
import threading
from dotenv import load_dotenv

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

    def capture_audio(self, chunk_size=1024, rate=44100):
        """
        Captures real-time audio from the microphone and saves it to a file.
        The recording will stop when the user presses Enter.
        
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
        
        # Start recording in a separate thread so we can listen for Enter to stop
        def record_audio():
            print("Recording... Press Enter to stop.")
            while True:
                data = stream.read(chunk_size)
                frames.append(data)
                if stop_recording_event.is_set():
                    break

        stop_recording_event = threading.Event()  # Event to control stopping the recording
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()

        # Wait for user to press Enter to stop the recording
        input("Press Enter to stop the recording...\n")
        stop_recording_event.set()  # Signal to stop recording

        # Clean up and save the audio
        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(audio_format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        recording_thread.join()  # Ensure the recording thread finishes
        logging.info(f"Recording saved to {output_file}")
        return output_file

    def split_audio(self, audio_file, chunk_duration=300):
        """
        Splits an audio file into smaller chunks of a specified duration.
        
        Args:
            audio_file (str): Path to the input audio file.
            chunk_duration (int): Duration of each chunk in seconds.
        
        Returns:
            list: Paths to the split audio chunks.
        """
        audio_chunks = []
        try:
            with wave.open(audio_file, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()

                chunk_frames = chunk_duration * framerate
                total_chunks = n_frames // chunk_frames + (n_frames % chunk_frames > 0)
                
                output_folder = os.path.dirname(audio_file)
                for i in range(total_chunks):
                    chunk_file = os.path.join(output_folder, f"chunk_{i+1}.wav")
                    with wave.open(chunk_file, 'wb') as chunk_wf:
                        chunk_wf.setnchannels(n_channels)
                        chunk_wf.setsampwidth(sampwidth)
                        chunk_wf.setframerate(framerate)
                        chunk_wf.writeframes(wf.readframes(chunk_frames))
                    audio_chunks.append(chunk_file)
        except Exception as e:
            logging.error(f"Error splitting audio: {e}")
        return audio_chunks

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

    def transcribe_long_audio(self, audio_file):
        """
        Transcribes a long audio file by splitting it into smaller chunks.
        
        Args:
            audio_file (str): Path to the long audio file.
        
        Returns:
            str: Combined transcription of all chunks.
        """
        audio_chunks = self.split_audio(audio_file)
        full_transcription = []
        for chunk in audio_chunks:
            transcription = self.transcribe_audio(chunk)
            if transcription:
                full_transcription.append(transcription)
        return " ".join(full_transcription)

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

def save_summary_to_json(summary, file_name=None):
    """
    Save the session summary to a JSON file.
    
    Args:
        summary (dict): The summary data to be saved.
        file_name (str): The name of the file to save the summary to (optional).
    """
    try:
        output_folder = os.path.join(OUTPUT_PATH, 'saved_summaries', summary["patient_name"])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if not file_name:
            file_name = f'{summary["patient_name"]}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_session_summary.json'
        file_path = os.path.join(output_folder, file_name)

        with open(file_path, "w") as json_file:
            json.dump(summary, json_file, indent=4)
        logging.info(f"Summary saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving summary to JSON: {e}")

if __name__ == "__main__":
    # Sample patient data (this would normally come from your application context)
    patient_data = {
        "name": "John Doe",
        "age": 30,
        "history": "No significant medical history."
    }

    # Your OpenAI API key (make sure to load it from your environment variable)
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize the AudioAgent
    audio_agent = AudioAgent(openai_api_key=openai_api_key, patient_data=patient_data)

    # Capture audio
    audio_file = audio_agent.capture_audio()

    # Transcribe the captured audio
    transcription = audio_agent.transcribe_long_audio(audio_file)
    logging.info(f"Transcription: {transcription}")

    # Summarize the transcription
    summary = audio_agent.summarize_transcript(transcription)
    logging.info(f"Session Summary: {summary['summary']}")

    # Save the summary to a JSON file
    save_summary_to_json(summary)
