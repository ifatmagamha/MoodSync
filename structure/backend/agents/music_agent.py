import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import sys

# Search query per SessionReport.environment_mood. Instrumental on purpose: lyrics compete with the conversation.
# ponytail: static map, replace with a learned recommender once sessions log listening feedback.
MOOD_QUERIES = {
    "calm": "calm piano instrumental",
    "uplifting": "uplifting acoustic instrumental",
    "grounding": "grounding meditation nature sounds",
    "neutral": "lofi instrumental focus",
}


class MusicAgent:
    def __init__(self, spotify_client_id, spotify_client_secret, redirect_uri):
        """
        Initialize the MusicAgent with Spotify credentials. Playback control needs a Premium account.
        """
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=spotify_client_id,
            client_secret=spotify_client_secret,
            redirect_uri=redirect_uri,
            scope="user-read-playback-state user-modify-playback-state"
        ))

    def find_playlist(self, query):
        """Return the first playlist matching the query, or None."""
        items = self.sp.search(q=query, limit=5, type="playlist")["playlists"]["items"]
        return next((p for p in items if p), None)  # Spotify returns null entries for removed playlists

    def play_for_mood(self, mood, query=None):
        """
        Play a playlist for the mood, or for the therapist's own query.

        Returns:
            tuple: (playlist dict, name of the device it plays on)
        """
        playlist = self.find_playlist(query or MOOD_QUERIES.get(mood, MOOD_QUERIES["neutral"]))
        if not playlist:
            raise RuntimeError("No playlist found on Spotify for this query.")
        devices = self.sp.devices()["devices"]
        if not devices:
            raise RuntimeError("No Spotify device found. Open Spotify on the room's speaker or computer first.")
        device = next((d for d in devices if d["is_active"]), devices[0])
        self.sp.start_playback(device_id=device["id"], context_uri=playlist["uri"])
        return playlist, device["name"]


if __name__ == "__main__":
    # Usage: python music_agent.py calm
    load_dotenv()
    music_agent = MusicAgent(os.getenv("SPOTIFY_CLIENT_ID"), os.getenv("SPOTIFY_CLIENT_SECRET"), os.getenv("REDIRECT_URI"))
    playlist, device = music_agent.play_for_mood(sys.argv[1] if len(sys.argv) > 1 else "calm")
    print(f"Playing {playlist['name']} on {device}")
