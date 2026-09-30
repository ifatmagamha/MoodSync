from backend import records

class TherapySessionManager:
    def __init__(self, audio_agent):
        self.audio_agent = audio_agent

    def process_audio(self, audio_file):
        """Transcribe, write the report with the previous session as context, save it, and return the record."""
        name = self.audio_agent.patient_data["name"]
        history = records.load_records(name)
        transcript = self.audio_agent.transcribe(audio_file)
        report = self.audio_agent.generate_report(transcript, history[-1]["report"] if history else None)
        return records.save_record(name, transcript, report.model_dump())
