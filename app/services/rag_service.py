from ollama import Client as OllamaClient


class RagService:
    def __init__(self, client: OllamaClient, embed_model: str):
        self.client = client
        self.embed_model = embed_model

    def embed(self, text: str) -> list[float]:
        response = self.client.embed(model=self.embed_model, input=text)
        return response.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(model=self.embed_model, input=texts)
        return response.embeddings
