from anthropic import Anthropic
from app.services.rag_service import RagService

class ChatService:
    def __init__(self, client: Anthropic, rag_service: RagService):
        self.client = client
        self.rag_service = rag_service

    def generate(self, prompt: str) -> str: 
        
        self.rag_service.say_something()  # Example of inter-service communication
               
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text