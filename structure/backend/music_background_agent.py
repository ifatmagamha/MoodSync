import spotipy
from spotipy.oauth2 import SpotifyOAuth

class MusicBackgroundAgent:
    def __init__(self, client_id, client_secret, redirect_uri):
        self.spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope="user-modify-playback-state"
        ))

    def curate_music(self, mood):
        """
        Curate and play music based on the detected mood.
        """
        mood_to_playlist = {
            "calm": "calm_playlist_id",
            "uplifting": "uplifting_playlist_id",
            "neutral": "neutral_playlist_id"
        }
        playlist_id = mood_to_playlist.get(mood, "neutral_playlist_id")
        try:
            self.spotify.start_playback(context_uri=f"spotify:playlist:{playlist_id}")
        except Exception as e:
            print(f"Error curating music: {e}")
