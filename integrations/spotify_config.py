import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SpotifyConfig:
    def __init__(self):
        """
        Initialize the Spotify API client with the provided client ID and client secret.
        """
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify Client ID and Secret must be provided in the environment variables.")

        # Set up the client credentials manager
        self.credentials_manager = SpotifyClientCredentials(
            client_id=self.client_id,
            client_secret=self.client_secret
        )
        
        # Create the Spotify API client
        self.sp = spotipy.Spotify(auth_manager=self.credentials_manager)

    def search_tracks(self, query, limit=10):
        """
        Search for tracks on Spotify using a query.
        
        Args:
            query (str): The search query.
            limit (int): The number of results to return.
        
        Returns:
            list: A list of track names and URIs.
        """
        result = self.sp.search(q=query, limit=limit, type="track")
        tracks = result["tracks"]["items"]
        track_list = [{"name": track["name"], "uri": track["uri"]} for track in tracks]
        return track_list

    def play_track(self, track_uri):
        """
        Play a track on the default Spotify device.
        
        Args:
            track_uri (str): The URI of the track to play.
        """
        self.sp.start_playback(uris=[track_uri])

    def get_playlist(self, playlist_id):
        """
        Retrieve a playlist from Spotify by ID.
        
        Args:
            playlist_id (str): The ID of the playlist to retrieve.
        
        Returns:
            dict: The playlist details.
        """
        playlist = self.sp.playlist(playlist_id)
        return playlist
