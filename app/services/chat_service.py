from anthropic import Anthropic

class ChatService:
    def __init__(self, client: Anthropic):
        self.client = client

    def generate(self, prompt: str) -> str:
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text