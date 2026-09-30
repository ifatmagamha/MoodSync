import streamlit as st
import pandas as pd
import tempfile
import json
import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend import records
from backend.agents.audio_agent import AudioAgent
from backend.agents.music_agent import MOOD_QUERIES, MusicAgent
from backend.agents.visual_environment_agent import LIGHT_SCENES, VisualLightAgent
from backend.orchestrator import TherapySessionManager

load_dotenv()

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
patients = records.list_patients()
choice = st.sidebar.selectbox("Patient", [p["name"] for p in patients] + ["➕ New patient"])
saved = next((p for p in patients if p["name"] == choice), {})
profile = {
    "name": st.sidebar.text_input("Patient Name", value=saved.get("name", "")),
    "age": st.sidebar.number_input("Age", value=saved.get("age", 30), step=1),
    "history": st.sidebar.text_area("History", value=saved.get("history", "")),
    "instructions": st.sidebar.text_area("Instructions for the assistant", value=saved.get("instructions", ""),
                                         placeholder="e.g. CBT for panic disorder; track avoidance and breathing practice"),
}
patient_name = profile["name"]
if st.sidebar.button("Save patient"):
    try:
        records.save_profile(profile)
        st.rerun()
    except ValueError as e:
        st.sidebar.error(str(e))

# Sidebar for room settings
st.sidebar.header("Room Settings")
lights_on = st.sidebar.toggle("Control Hue lights")
bridge_ip = st.sidebar.text_input("Hue bridge IP", placeholder="empty = preview only", disabled=not lights_on)
music_on = st.sidebar.toggle("Control Spotify")


@st.cache_resource
def music_agent():
    return MusicAgent(os.getenv("SPOTIFY_CLIENT_ID"), os.getenv("SPOTIFY_CLIENT_SECRET"), os.getenv("REDIRECT_URI"))


@st.cache_resource
def light_agent(ip):
    return VisualLightAgent(ip or None)


def render_report(record, key):
    r = record["report"]
    for f in r["risk_flags"]:
        st.error(f"**Risk — {f['type'].replace('_', ' ')} ({f['severity']})**: “{f['evidence']}”")
    st.markdown(f"**{record['timestamp']}** — {r['overview']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mood", f"{r['scores']['mood']}/10")
    c2.metric("Anxiety", f"{r['scores']['anxiety']}/10")
    c3.metric("Engagement", f"{r['scores']['engagement']}/10")
    st.markdown("**Emotions**: " + ", ".join(f"{e['name']} ({e['intensity']}/5)" for e in r["emotions"]))
    st.markdown(f"**Progress since last session**: {r['progress_since_last_session']}")
    for title, field in (("Themes", "themes"), ("Key insights", "key_insights"),
                         ("Therapeutic goals", "therapeutic_goals"), ("Homework", "homework"),
                         ("Evidence", "evidence_quotes")):
        if r[field]:
            st.markdown(f"**{title}**\n" + "\n".join(f"- {item}" for item in r[field]))
    with st.expander("Transcript"):
        st.write(record["transcript"])
    st.download_button("Download report (JSON)", json.dumps(record, indent=2, ensure_ascii=False), key=key,
                       file_name=f"{records.patient_id(record['patient_name'])}_{record['timestamp'][:10]}.json",
                       mime="application/json")


history = records.load_records(patient_name) if patient_name.strip() else []
session_tab, history_tab, room_tab = st.tabs(["🎙️ Session", "📈 Patient history", "💡 Room"])

with session_tab:
    consent = st.checkbox("The patient consented to this session being recorded and analysed")
    audio = st.audio_input("Record the session", disabled=not consent)
    uploaded = st.file_uploader("…or upload a WAV recording", type=["wav"], disabled=not consent)
    source = audio or uploaded
    if source and st.button("Generate session report", type="primary"):
        # Audio is only kept long enough to transcribe; the transcript is what gets stored.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(source.getvalue())
        try:
            with st.spinner("Transcribing and writing the report..."):
                records.save_profile(profile)
                agent = AudioAgent(profile)
                st.session_state.record = TherapySessionManager(agent).process_audio(tmp.name)
            st.rerun()  # refresh history and the room suggestion
        except Exception as e:
            st.error(f"Processing failed: {e}")
        finally:
            os.remove(tmp.name)
    if "record" in st.session_state:
        render_report(st.session_state.record, key="session")

with history_tab:
    if not history:
        st.info("No sessions recorded for this patient yet.")
    else:
        for alert in records.weak_signals(history):
            st.warning(alert)
        st.line_chart(pd.DataFrame([r["report"]["scores"] for r in history], index=[r["timestamp"] for r in history]))
        chosen = st.selectbox("Session", history[::-1], format_func=lambda r: r["timestamp"])
        render_report(chosen, key="history")

with room_tab:
    moods = list(LIGHT_SCENES)
    suggested = history[-1]["report"]["environment_mood"] if history else "neutral"
    mood = st.radio("Ambience", moods, index=moods.index(suggested), horizontal=True,
                    help="Pre-selected from the latest session report")
    scene = LIGHT_SCENES[mood]
    st.markdown(f'<div style="height:80px;border-radius:12px;background:{scene["hex"]}"></div>', unsafe_allow_html=True)
    query = st.text_input("Music search", value=MOOD_QUERIES[mood])
    if st.button("Apply to room"):
        try:
            light_agent(bridge_ip if lights_on else "").adjust_lighting(mood)
            st.success("Lights updated." if lights_on and bridge_ip else "Light preview only (no bridge connected).")
            if music_on:
                playlist, device = music_agent().play_for_mood(mood, query)
                st.success(f"Playing “{playlist['name']}” on {device}.")
        except Exception as e:
            st.error(f"Could not update the room: {e}")
