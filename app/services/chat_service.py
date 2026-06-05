from anthropic import Anthropic
from app.services.rag_service import RagService
from app.models.chat_schema import ChatResponse

class ChatService:
    def __init__(self, client: Anthropic, rag_service: RagService):
        self.client = client
        self.rag_service = rag_service

    def generate(self, prompt: str) -> str: 
        prompt_vector = self.rag_service.embed(prompt)
        retrieved_data = self.rag_service.vector_store.search(prompt_vector)

        content = [item["content"] for item in retrieved_data]
        augmented_prompt = f'''
         
         Prompt: {prompt}
         
         Documentation: {content}
        
        '''
        response = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system='''
            You are a helpful assistant that provides legal necessary information about import-export on merchandise based on retrieved document chunks.
            If the retrieved chunks do not contain relevant information, respond with "I don't know". Always use the retrieved information to answer the user's question. Do not make up any information.
            Follow the output format strictly from the ChatResponse schema.
            ''',
            messages=[{"role": "user", "content": augmented_prompt}],
            output_format=ChatResponse
        )
        
        return response.parsed_output