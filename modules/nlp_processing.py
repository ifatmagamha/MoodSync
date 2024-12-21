class NLPProcessor:
    def analyze_emotion(self, text):
        """Analyze the mood or emotion of the conversation"""
        # For simplicity, let's assume we classify the mood based on keywords
        if "happy" in text:
            return "happy"
        elif "sad" in text:
            return "sad"
        elif "angry" in text:
            return "angry"
        else:
            return "relaxed"

    def summarize_conversation(self, text):
        """Generate bullet points from the conversation"""
        bullet_points = text.split(".")
        return [point.strip() for point in bullet_points if point]
