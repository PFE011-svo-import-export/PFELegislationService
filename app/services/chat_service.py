import time
from anthropic import Anthropic
from app.services.rag_service import RagService
from app.models.TraitementTarifiaire import TraitementTarifiare
from app.models.ExigencesImportation import ExigencesImportation
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
class ChatService:
    def __init__(self, client: Anthropic, rag_service: RagService):
        self.client = client
        self.rag_service = rag_service
    
    def search_traitement_tarifiaires(self, pays: str) -> TraitementTarifiare:
        prompt = f"Traitements tarifaires applicables au {pays}?"
        return self.generate(prompt, TraitementTarifiare)
    
    def search_exigences_importation(self, produit: str, pays_importation: str) -> ExigencesImportation:
        prompt = f"Selon le {produit} et le {pays_importation}, donne moi la  liste des exigences que l'exportateur doit respecter pour l'import de produit au canada concernant les sujets suivants : Emballage de Bois (Normes NIMP15), Justification de l'origine, Marquage / Étiquetage, Exigences de Bilinguisme, Exigences, d'importation générales, Certification Biologique, Exigences Canadiennes sur la Salubrité (RSAC)"
        return self.generate(prompt, ExigencesImportation)

    def generate(self, prompt: str, output_model: Type[T]) -> T:
        retrieval_start = time.perf_counter()
        candidates = self.rag_service.retrieve(prompt)
        retrieval_elapsed = time.perf_counter() - retrieval_start
        print(f"Retrieval took {retrieval_elapsed:.2f}s")
        print(f"Here are the candidates retrieved from the vector store:\n{candidates}")

        count = self.client.messages.count_tokens(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": prompt}]
        )

        print(f"Total prompt tokens: {count.input_tokens}")
        
        documentation = "\n\n".join(
            f"[Source: {c['source']}]\n{c['content']}"
            for c in candidates
        )

        augmented_prompt = f'''

         Prompt: {prompt}

         Documentation: {documentation}

        '''
        llm_start = time.perf_counter()
        response = self.client.messages.parse(
            model="claude-sonnet-4-6",
            max_tokens=10000,
            output_config={"effort": "low"},
            system='''
            You are a helpful assistant that provides legal necessary information about import-export on merchandise based on retrieved document chunks.
            If the retrieved chunks do not contain relevant information, use empty strings for the fields. Always use the retrieved information to answer the user's question. Do not make up any information.
            ''',
            messages=[{"role": "user", "content": augmented_prompt}],
            output_format=output_model,
        )
        llm_elapsed = time.perf_counter() - llm_start

        
        print(f"Total input tokens: {response.usage.input_tokens}")
        print(f"Total output tokens: {response.usage.output_tokens}")

        print(f"LLM generation took {llm_elapsed:.2f}s")

        return response.parsed_output

    def answer_prompt(self, prompt: str) -> dict:
        """Répond à une question libre de l'utilisateur en langage naturel,
        en se basant sur les chunks récupérés. Utilisé par l'interface de test."""

        candidates = self.rag_service.retrieve(prompt)

        documentation = "\n\n".join(
            f"[Source: {c['source']}]\n{c['content']}"
            for c in candidates
        )

        augmented_prompt = f"Question: {prompt}\n\nDocumentation:\n{documentation}"

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            output_config={"effort": "low"},
            system='''
            Tu es un assistant spécialisé en législation d'import-export de marchandises au Canada.
            Réponds à la question de l'utilisateur en te basant UNIQUEMENT sur la documentation fournie.
            Si la documentation ne contient pas l'information nécessaire, dis-le clairement et n'invente rien.
            Réponds en français, de manière claire et concise.
            ''',
            messages=[{"role": "user", "content": augmented_prompt}],
        )

        answer = next((block.text for block in response.content if block.type == "text"), "")
        # Sources uniques, dans l'ordre de pertinence renvoyé par le reranker.
        sources = list(dict.fromkeys(c["source"] for c in candidates))

        return {"answer": answer, "sources": sources}