class ImageGenerator:
    def __init__(self):
        from integrations.openai_config import OpenAIConfig
        self.openai = OpenAIConfig()

    def generate_image(self, mood):
        """Generate therapeutic images based on the detected mood"""
        prompt = f"Generate a therapeutic image for someone feeling {mood}"
        image = self.openai.generate_mood_based_message(prompt)
        print(f"Generated image: {image}")
        return image
