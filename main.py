from modules.speech_to_text import SpeechToText
from integrations.openai_config import OpenAIConfig

def main():
    """Main function to run the psychology AI tool."""
    print("Initializing Psychology AI Tool...")

    # Initialize OpenAI configuration
    openai_config = OpenAIConfig()  # Create an OpenAIConfig instance
    
    # Initialize SpeechToText with the OpenAIConfig instance
    speech_to_text = SpeechToText(openai_config)

    # Provide the correct path to your audio file
    audio_file_path = "path_to_audio_file.wav"  # Replace with your actual audio file path

    # Step 1: Real-time Speech-to-Text
    print("Starting real-time transcription...")
    live_transcription = speech_to_text.transcribe_audio(audio_file_path)

    if live_transcription:
        print("Transcription completed successfully:")
        print(live_transcription)
    else:
        print("Error: No transcription available.")

if __name__ == "__main__":
    main()
