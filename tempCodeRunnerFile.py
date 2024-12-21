import sys
import os

# Add the root project directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from modules.speech_to_text import SpeechToText
from modules.nlp_processing import NLPProcessor
from modules.mood_music import MoodMusic
from modules.image_generation import ImageGenerator

def main():
    """Main function to run the psychology AI tool."""
    print("Initializing Psychology AI Tool...")

    # Initialize components
    speech_to_text = SpeechToText()
    nlp_processor = NLPProcessor()
    mood_music = MoodMusic()
    image_generator = ImageGenerator()

    # Step 1: Real-time Speech-to-Text
    print("Starting real-time transcription...")
    live_transcription = speech_to_text.transcribe_audio()

    # Step 2: NLP Processing
    print("Processing transcription for bullet points and emotion analysis...")
    bullet_points = nlp_processor.summarize_conversation(live_transcription)
    detected_emotion = nlp_processor.analyze_emotion(live_transcription)

    # Step 3: Music Recommendation
    print("Recommending music based on detected emotion...")
    mood_music.recommend_and_play_music(detected_emotion)

    # Step 4: Image Generation
    print("Generating therapeutic image...")
    therapeutic_image = image_generator.generate_image(detected_emotion)

    # Step 5: Display results
    print(f"Bullet points: {bullet_points}")
    print(f"Therapeutic image: {therapeutic_image}")

if __name__ == "__main__":
    main()
