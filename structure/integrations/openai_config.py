import os
from openai import OpenAI

class OpenAIConfig:
    def __init__(self, api_key=None):
        """
        Initialize the OpenAI client with the provided API key or an environment variable.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key for OpenAI must be provided or set as an environment variable.")
        self.client = OpenAI(api_key=self.api_key)

    def create_chat_completion(self, model, messages, **kwargs):
        """
        Create a chat completion using the specified model and messages.
        
        Args:
            model (str): The model to use (e.g., 'gpt-4', 'gpt-3.5-turbo').
            messages (list): A list of message dictionaries (role and content).
            **kwargs: Additional parameters for the completion (e.g., temperature, max_tokens).
            
        Returns:
            dict: The response from the OpenAI API.
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
