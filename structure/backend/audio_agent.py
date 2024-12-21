import openai
import pyaudio
import wave

class AudioAgent:
    def __init__(self, openai_api_key, patient_data, target_language="en"):
        self.openai_api_key = openai_api_key
        self.patient_data = patient_data
        self.target_language = target_language
        openai.api_key = self.openai_api_key

    def capture_audio(self, chunk_size=1024, rate=44100, duration=1800):
        audio_format = pyaudio.paInt16
        channels = 1
        output_file = "recorded_audio.wav"

        audio = pyaudio.PyAudio()
        stream = audio.open(format=audio_format, channels=channels, 
                            rate=rate, input=True, frames_per_buffer=chunk_size)

        print("Recording...")
        frames = []
        for _ in range(0, int(rate / chunk_size * duration)):
            data = stream.read(chunk_size)
            frames.append(data)
        print("Recording complete.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(audio_format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        return output_file

    def transcribe_audio(self, audio_file):
        try:
            print("Transcribing audio...")
            with open(audio_file, "rb") as file:
                response = openai.Audio.transcribe(
                    model="whisper-1",
                    file=file,
                    language=None  
                )
            return response["text"]
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None

    def translate_text(self, text):
        try:
            print(f"Translating text to {self.target_language}...")
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Translate the following text to {self.target_language}: {text}",
                max_tokens=200
            )
            translated_text = response.choices[0].text.strip()
            return translated_text
        except Exception as e:
            print(f"Error during translation: {e}")
            return None

    def summarize_transcript(self, transcript):
        patient_context = f"Patient Name: {self.patient_data.get('name', 'Unknown')}, " \
                          f"Age: {self.patient_data.get('age', 'Unknown')}, " \
                          f"History: {self.patient_data.get('history', 'No history provided')}."
        
        prompt = (f"Here is the context for the patient:\n{patient_context}\n\n"
                  f"Now summarize the following transcript into key insights and "
                  f"actionable points for a therapy session:\n\n{transcript}")
        
        try:
            print("Generating summary...")
            response = openai.ChatCompletion.create(
                model="gpt-4", 
                messages=[
                    {"role": "system", "content": "You are an assistant helping a therapist interpret patient speech."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['choices'][0]['message']['content'] if response else None
        except Exception as e:
            print(f"Error generating summary: {e}")
            return None

    def transcribe_and_translate_audio(self, audio_file):
        transcript = self.transcribe_audio(audio_file)
        
        if transcript:
            if self.target_language != "en":
                translated_text = self.translate_text(transcript)
                return translated_text
            else:
                return transcript
        else:
            return "Transcription failed."

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    patient_data = {
        "name": "John Doe",
        "age": 32,
        "history": "Anxiety and mild depression."
    }

    audio_agent = AudioAgent(openai_api_key=openai_api_key, patient_data=patient_data, target_language="en")

    audio_file = audio_agent.capture_audio(duration=10)
    translated_text = audio_agent.transcribe_and_translate_audio(audio_file)
    print(f"Translated Text: {translated_text}")

    if translated_text:
        summary = audio_agent.summarize_transcript(translated_text)
        print("\nSummary:")
        print(summary)
    else:
        print("No transcript available.")
