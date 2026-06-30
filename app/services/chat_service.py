from anthropic import Anthropic
from app.services.rag_service import RagService
from app.models.chat_schema import TraitementTarifiare

class ChatService:
    def __init__(self, client: Anthropic, rag_service: RagService):
        self.client = client
        self.rag_service = rag_service

    def generate(self, prompt: str) -> TraitementTarifiare:
        prompt_vector = self.rag_service.embed(prompt)
        retrieved_data = self.rag_service.vector_store.search(prompt_vector)

        for item in retrieved_data:
            print(f"[RAG] source={item['source']} | section={item['heading_path']} | score={item['score']:.4f}")
        content = [item["content"] for item in retrieved_data]
        augmented_prompt = f'''

         Prompt: {prompt}

         Documentation: {content}

        '''
        # response = self.client.messages.parse(
        #     model="claude-sonnet-4-6",
        #     max_tokens=4096,
        #     system='''
        #     You are a helpful assistant that provides legal necessary information about import-export on merchandise based on retrieved document chunks.
        #     If the retrieved chunks do not contain relevant information, use empty strings for the fields. Always use the retrieved information to answer the user's question. Do not make up any information.
        #     ''',
        #     messages=[{"role": "user", "content": augmented_prompt}],
        #     output_format=TraitementTarifiare,
        # )

        response = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=10000,
            system='''
            You are a helpful assistant that provides legal necessary information about import-export on merchandise based on retrieved document chunks.
            If the retrieved chunks do not contain relevant information, use empty strings for the fields. Always use the retrieved information to answer the user's question. Do not make up any information.
            For this question, there should be 3 traitement_tarifiaire_applicable.
            ''',
            messages=[{"role": "user", "content": augmented_prompt}],
            output_format=TraitementTarifiare,
        )

        return response.parsed_output