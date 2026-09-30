"""Offline checks (no API keys needed). Run: python structure/test_moodsync.py"""
import tempfile
import wave
from types import SimpleNamespace
from typing import get_args

from openai.lib._pydantic import to_strict_json_schema

from backend import records
from backend.agents.audio_agent import AudioAgent, SessionReport, split_audio
from backend.agents.music_agent import MOOD_QUERIES
from backend.agents.visual_environment_agent import LIGHT_SCENES, VisualLightAgent


def report(mood=5, anxiety=5, engagement=5, flags=()):
    return {"scores": {"mood": mood, "anxiety": anxiety, "engagement": engagement}, "risk_flags": list(flags)}


def main():
    # Names from the dashboard can't escape the records folder.
    assert records.patient_id("../../Khlifa KHLIF") == "khlifa_khlif"
    try:
        records.patient_id(" ../ ")
        raise AssertionError("empty id accepted")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory() as root:
        records.save_record("Jane", "hello", report(), root=root)
        assert records.load_records("Jane", root=root)[0]["transcript"] == "hello"
        assert records.load_records("Nobody", root=root) == []
        records.save_profile({"name": "Jane", "instructions": "CBT"}, root=root)
        assert records.list_patients(root=root) == [{"name": "Jane", "instructions": "CBT"}]

    # Weak signals.
    rec = lambda **kw: {"report": report(**kw)}
    assert records.weak_signals([]) == []
    assert records.weak_signals([rec(), rec(), rec()]) == []
    assert any("Mood 3/10" in a for a in records.weak_signals([rec(mood=6), rec(mood=6), rec(mood=3)]))
    assert any("Anxiety" in a for a in records.weak_signals([rec(anxiety=3), rec(anxiety=6)]))
    assert any("three sessions" in a for a in records.weak_signals([rec(mood=7), rec(mood=6), rec(mood=5)]))
    flag = {"type": "self_harm", "severity": "high", "evidence": "I don't want to be here"}
    assert any("self harm" in a for a in records.weak_signals([rec(flags=[flag])]))
    assert records.weak_signals([rec(flags=[dict(flag, severity="low")])]) == []

    # Chunks stay under the size limit and lose no audio.
    with tempfile.TemporaryDirectory() as tmp:
        src = f"{tmp}/in.wav"
        with wave.open(src, "wb") as wf:
            wf.setnchannels(1), wf.setsampwidth(2), wf.setframerate(8000)
            wf.writeframes(b"\x01\x00" * 8000)  # 1 s = 16000 bytes
        chunks = split_audio(src, tmp, max_bytes=6000)
        assert len(chunks) == 3
        assert sum(wave.open(c).getnframes() for c in chunks) == 8000

    # OpenAI structured outputs accepts the schema, and every mood the model can pick has music and light.
    to_strict_json_schema(SessionReport)
    moods = set(get_args(SessionReport.model_fields["environment_mood"].annotation))
    assert moods == set(MOOD_QUERIES) == set(LIGHT_SCENES)
    assert VisualLightAgent().adjust_lighting("unknown") == LIGHT_SCENES["neutral"]

    # generate_report sends the previous session and the therapist's instructions, and returns the parsed report.
    sent = {}
    parsed = SessionReport.model_construct(overview="ok")
    def parse(**kw):
        sent.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    agent = AudioAgent({"name": "Jane", "instructions": "track avoidance"}, client=client)
    assert agent.generate_report("I feel better", {"overview": "last week"}) is parsed
    user_msg = sent["messages"][1]["content"]
    assert "last week" in user_msg and "track avoidance" in user_msg and "I feel better" in user_msg
    assert sent["response_format"] is SessionReport

    print("all checks passed")


if __name__ == "__main__":
    main()
