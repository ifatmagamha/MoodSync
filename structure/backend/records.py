"""
Patient storage, one folder per patient:

    saved_outputs/patients/<patient_id>/profile.json          name, age, history, therapist instructions
    saved_outputs/patients/<patient_id>/sessions/<ts>.json    transcript + SessionReport

ponytail: JSON files for a single-therapist prototype; move to Postgres (encrypted) before a second therapist.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

# Anchored to the repo, not the working directory (the old relative default produced two saved_outputs folders).
RECORDS_DIR = Path(os.getenv("OUTPUT_PATH", Path(__file__).resolve().parents[1] / "saved_outputs")) / "patients"


def patient_id(name):
    """Folder-safe id. The name is free text from the dashboard, so it must never reach the filesystem as-is."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("A patient name is required.")
    return slug


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_profile(profile, root=RECORDS_DIR):
    """profile: dict with name, age, history, instructions."""
    _write(Path(root) / patient_id(profile["name"]) / "profile.json", profile)


def list_patients(root=RECORDS_DIR):
    """All patient profiles, sorted by id."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(root).glob("*/profile.json"))]


def save_record(patient_name, transcript, report, root=RECORDS_DIR):
    """Save one session. `report` is a SessionReport dumped to a dict."""
    now = datetime.now()
    record = {"patient_name": patient_name, "timestamp": now.isoformat(timespec="seconds"),
              "transcript": transcript, "report": report}
    _write(Path(root) / patient_id(patient_name) / "sessions" / f"{now:%Y%m%d_%H%M%S}.json", record)
    return record


def load_records(patient_name, root=RECORDS_DIR):
    """All sessions for a patient, oldest first."""
    folder = Path(root) / patient_id(patient_name) / "sessions"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]


def weak_signals(records, window=3, threshold=2):
    """Plain-language alerts comparing the latest session with the patient's recent baseline."""
    if not records:
        return []
    latest = records[-1]["report"]
    alerts = [f"Risk flag ({f['severity']}): {f['type'].replace('_', ' ')} — “{f['evidence']}”"
              for f in latest["risk_flags"] if f["severity"] != "low"]

    baseline = [r["report"]["scores"] for r in records[-1 - window:-1]]
    for metric, worse_if_higher in (("mood", False), ("anxiety", True), ("engagement", False)):
        if not baseline:
            break
        mean = sum(s[metric] for s in baseline) / len(baseline)
        delta = latest["scores"][metric] - mean
        if (delta if worse_if_higher else -delta) >= threshold:
            alerts.append(f"{metric.capitalize()} {latest['scores'][metric]}/10 vs recent average {mean:.1f}")

    # A slow drift never crosses the threshold in one step, so check the direction too.
    moods = [r["report"]["scores"]["mood"] for r in records[-3:]]
    if len(moods) == 3 and moods[0] > moods[1] > moods[2]:
        alerts.append("Mood has dropped three sessions in a row")
    return alerts
