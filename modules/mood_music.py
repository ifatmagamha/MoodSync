class MoodMusic:
    def __init__(self):
        from integrations.spotify_config import SpotifyConfig
        self.spotify = SpotifyConfig()

    def recommend_and_play_music(self, mood):
        """Recommend and play music based on mood"""
        tracks = self.spotify.recommend_music(mood)
        print(f"Recommended tracks: {', '.join(tracks)}")
        self.spotify.play_music(tracks[0])  # Play the first recommended track
