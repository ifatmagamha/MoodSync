"""Generate a synthetic two-voice session WAV for demos and tests, so no real patient audio is ever needed.
Run: python structure/make_demo_session.py   -> structure/saved_outputs/demo_session.wav
"""
import wave
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

DIALOGUE = [
    ("sage", "Welcome back. How has your week been since we last talked about the panic attacks?"),
    ("ash", "Honestly, a bit better. I tried the breathing exercise twice when I felt my chest getting tight, and it helped."),
    ("sage", "That's great progress. What was different about those moments?"),
    ("ash", "I noticed it earlier. But work is still overwhelming, and I've been avoiding the team meetings."),
    ("sage", "What goes through your mind before a meeting?"),
    ("ash", "That everyone will see I'm not good enough. Some nights I think everyone would be better off without me."),
    ("sage", "Thank you for telling me that. I want to come back to it carefully. Are you having thoughts of hurting yourself?"),
    ("ash", "No, not like that. It's more that I feel like a burden. Sleeping has been hard too."),
    ("sage", "Let's keep practising the breathing daily, and this week try attending one meeting for ten minutes."),
]

if __name__ == "__main__":
    load_dotenv()
    client = OpenAI()
    out = Path(__file__).parent / "saved_outputs" / "demo_session.wav"
    out.parent.mkdir(exist_ok=True)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1), wf.setsampwidth(2), wf.setframerate(24000)  # the API's raw pcm format
        for voice, text in DIALOGUE:
            pcm = client.audio.speech.create(model="gpt-4o-mini-tts", voice=voice, input=text, response_format="pcm")
            wf.writeframes(pcm.content + b"\x00\x00" * 12000)  # half a second of silence between turns
    print(f"Wrote {out}")
