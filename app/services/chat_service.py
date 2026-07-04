from anthropic import Anthropic
from app.services.rag_service import RagService
from app.models.TraitementTarifiaire import TraitementTarifiare
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
class ChatService:
    def __init__(self, client: Anthropic, rag_service: RagService):
        self.client = client
        self.rag_service = rag_service
    

    def search_traitement_tarifiaires(self, pays: str) -> TraitementTarifiare:
        prompt = f"Traitements tarifiaires applicables au {pays}?"
        #prompt = f"Quels traitements tarifaires (NPF, TPG, TPMD, Autres) sont accordés au pays {pays}?"
        return self.generate(prompt, TraitementTarifiare)

    def generate(self, prompt: str, output_model: Type[T]) -> T:
        print(f"Recieved prompt: {prompt}")
        candidates = self.rag_service.retrieve(prompt)

        documentation = "\n\n".join(
            f"[Source: {c['source']}]\n{c['content']}"
            for c in candidates
        )

        print(f"candidates: {documentation}")

        augmented_prompt = f'''

         Prompt: {prompt}

         Documentation: {documentation}

        '''
        response = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=10000,
            system='''
            You are a helpful assistant that provides legal necessary information about import-export on merchandise based on retrieved document chunks.
            If the retrieved chunks do not contain relevant information, use empty strings for the fields. Always use the retrieved information to answer the user's question. Do not make up any information.
            ''',
            messages=[{"role": "user", "content": augmented_prompt}],
            output_format=output_model,
        )

        return response.parsed_output