from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from app.models.chunk_schema import DocumentChunk

class VectorStore:
    COLLECTION_NAME = "documents-legislatives-import-export"
    VECTOR_SIZE = 768  # nomic-embed-text

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=qdrant_url)

    def ensure_collection(self):
        if not self.client.collection_exists(self.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def store(self, chunks: list[DocumentChunk]):
        points = [
            PointStruct(
                id=chunk.id,
                vector=chunk.vector,
                payload={
                    "content": chunk.content,
                    "source": chunk.metadata.source,
                    "heading_path": chunk.metadata.heading_path,
                    "chunk_index": chunk.metadata.chunk_index,
                }
            )
            for chunk in chunks
        ]
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points
        )

    def search(self, query_vector: list[float], top_k: int = 3) -> list[dict]:
        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )
        return [
            {
                "content": r.payload["content"],
                "heading_path": r.payload["heading_path"],
                "source": r.payload["source"],
                "score": r.score,
            }
            for r in results.points
        ]