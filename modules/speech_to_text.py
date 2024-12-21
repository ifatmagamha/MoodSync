import pyaudio
import wave
from integrations.openai_config import OpenAIConfig

class SpeechToText:
    def __init__(self):
        self.openai = OpenAIConfig()

    def transcribe_audio(self, duration=10):
        """Record audio for the given duration and transcribe it"""
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 1
        rate = 44100
        record_seconds = duration
        output_filename = "audio_output.wav"

        p = pyaudio.PyAudio()
        stream = p.open(format=sample_format, channels=channels,
                        rate=rate, input=True, frames_per_buffer=chunk)

        frames = []
        for _ in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save audio to file
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Transcribe audio using OpenAI
        with open(output_filename, "rb") as audio_file:
            transcript = self.openai.transcribe_audio(audio_file)
        return transcript
