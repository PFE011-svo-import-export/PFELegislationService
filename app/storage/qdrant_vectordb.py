from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, PayloadSchemaType,
    SparseVectorParams, SparseVector, Modifier, Prefetch, FusionQuery, Fusion,
)
from app.models.chunk_schema import DocumentChunk

class VectorStore:
    COLLECTION_NAME = "documents-legislatives-import-export"
    VECTOR_SIZE = 3072

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_api_key: str | None = None,
        qdrant_use_https: bool = False,
    ):
        self.client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            https=qdrant_use_https,
            api_key=qdrant_api_key or None,
        )

    def ensure_collection(self):
        if not self.client.collection_exists(self.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                # Vecteurs nommés : "dense" pour l'embedding sémantique, "bm25" pour le sparse
                vectors_config={
                    "dense": VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams(
                        modifier=Modifier.IDF,  # nécessaire pour un vrai scoring BM25 (sinon c'est juste du term frequency)
                    ),
                },
            )
        # Payload index on `source` is required to filter on it (is_source_ingested).
        # create_payload_index is idempotent, so it's safe to call on every startup.
        self.client.create_payload_index(
            collection_name=self.COLLECTION_NAME,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def store(self, chunks: list[DocumentChunk], batch_size: int = 128):
        points = [
            PointStruct(
                id=chunk.id,
                vector={
                    "dense": chunk.vector,
                    "bm25": SparseVector(
                        indices=chunk.sparse_vector["indices"],
                        values=chunk.sparse_vector["values"],
                    ),
                },
                payload={
                    "content": chunk.content,
                    "source": chunk.metadata.source,
                    "heading_path": chunk.metadata.heading_path,
                    "chunk_index": chunk.metadata.chunk_index,
                }
            )
            for chunk in chunks
        ]
        # Upsert par lots : un fichier peut produire des milliers de points, et une seule
        # requête trop volumineuse fait fermer la connexion côté Qdrant.
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points[start:start + batch_size],
            )

    def search(self, query_vector: list[float], top_k: int = 10) -> list[dict]:
        """Recherche dense uniquement — conservée pour compatibilité/debug."""
        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            using="dense",  # requis maintenant que les vecteurs sont nommés
            limit=top_k
        )
        return self._format_results(results.points)

    def hybrid_search(self, query_vector: list[float], query_sparse: dict, top_k: int = 5, prefetch_limit: int = 15) -> list[dict]:
        """Recherche hybride dense + BM25, fusionnée avec Reciprocal Rank Fusion."""
        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=query_sparse["indices"],
                        values=query_sparse["values"],
                    ),
                    using="bm25",
                    limit=prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )
        return self._format_results(results.points)

    def _format_results(self, points) -> list[dict]:
        return [
            {
                "content": r.payload["content"],
                "heading_path": r.payload["heading_path"],
                "source": r.payload["source"],
                "score": r.score,
            }
            for r in points
        ]

    def is_source_ingested(self, source: str) -> bool:
        results, _ = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(results) > 0

    def delete_collection(self):
        self.client.delete_collection(collection_name=self.COLLECTION_NAME)

    def ping(self) -> bool:
        """Vérifie que Qdrant est joignable. Utilisé par le probe de readiness."""
        self.client.get_collections()
        return True