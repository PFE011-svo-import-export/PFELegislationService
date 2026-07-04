from anthropic import Anthropic
from app.services.rag_service import RagService
from app.models.chat_schema import TraitementTarifiare

class ChatService:
    def __init__(self, client: Anthropic, rag_service: RagService):
        self.client = client
        self.rag_service = rag_service

    def generate(self, prompt: str) -> TraitementTarifiare:
        candidates = self.rag_service.retrieve(prompt)

        augmented_prompt = f'''

         Prompt: {prompt}

         Documentation: {candidates}

        '''
        response = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system='''
            You are a helpful assistant that provides legal necessary information about import-export on merchandise based on retrieved document chunks.
            If the retrieved chunks do not contain relevant information, use empty strings for the fields. Always use the retrieved information to answer the user's question. Do not make up any information.
            ''',
            messages=[{"role": "user", "content": augmented_prompt}],
            output_format=TraitementTarifiare,
        )

        return response.parsed_output

        return ""