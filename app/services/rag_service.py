
from app.helpers.semantic_chunker import SemanticChunker
from app.storage.qdrant_vectordb import VectorStore
import uuid
import os
from app.models.chunk_schema import ChunkMetadata, DocumentChunk
from fastembed import SparseTextEmbedding
from openai import OpenAI
import cohere

class RagService:
    def __init__(self, client: OpenAI, embed_model: str, vector_store: VectorStore, reranker_model: cohere.ClientV2):
        self.embedding_client = client
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.reranker = reranker_model
        self.reranker_model_name = "rerank-v4.0-fast"
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    # ─── Chunking + ingestion ─────────────────────────────────────────────────────
    def ingest_folder(self, folder_path: str) -> dict:
        self.vector_store.ensure_collection()
        ingested = []
        skipped = []

        md_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]

        for filename in md_files:
            if self.vector_store.is_source_ingested(filename):
                skipped.append(filename)
                print(f"Skipping already ingested file: {filename}")
                continue

            filepath = os.path.join(folder_path, filename)
            self.embed_md_file(filepath)
            ingested.append(filename)
            print(f"Ingested: {filename}")

        return {"ingested": ingested, "skipped": skipped}
    
    def embed_md_file(self, filepath: str) -> list[dict]:
        semantic_chunker = SemanticChunker()
        chunks = semantic_chunker.chunk_md_file(filepath)
        texts = [c["content"] for c in chunks]

        dense_vectors = self.embed_batch(texts)
        sparse_vectors = self.embed_sparse_batch(texts)
        
        chunks_to_store = [
            DocumentChunk(
                id=str(uuid.uuid4()),
                vector=dense_vec,
                sparse_vector=sparse_vec,  # nouveau champ à ajouter au modèle
                content=chunk["content"],
                metadata=ChunkMetadata(**chunk["metadata"])
            )
            for dense_vec, sparse_vec, chunk in zip(dense_vectors, sparse_vectors, chunks)
        ]
        
        self.vector_store.store(chunks_to_store)
        return chunks_to_store
    
    # ─── Retrieval + reranking ─────────────────────────────────────────────────────
    def retrieve(self, prompt: str) -> list[dict]:
        prompt_dense = self.embed(prompt)
        prompt_sparse = self.embed_sparse_batch([prompt])[0]

        initial_candidates = self.vector_store.hybrid_search(prompt_dense, prompt_sparse)
        #initial_candidates = self.vector_store.search(prompt_dense, 15)

        reranked = self.rerank_candidates(prompt, initial_candidates, topk=5)

        return reranked

    def rerank_candidates(self, prompt: str, initial_candidates: list[dict], topk: int) -> list[dict]:
        if not initial_candidates:
            return []
        # On fournit le heading_path au cross-encoder en plus du content : le contexte
        # de la section (ex. « Liste des Pays et Traitements Tarifaires... ») porte un
        # signal essentiel que le content, souvent très laconique, ne contient pas.
        texts = [f"{c['heading_path']}\n{c['content']}" for c in initial_candidates]

        response = self.reranker.rerank(
            model=self.reranker_model_name,
            query=prompt,
            documents=texts,
            top_n=topk,
        )

        # Cohere renvoie les résultats déjà triés par pertinence décroissante et
        # limités à top_n. Chaque résultat référence le candidat d'origine par son
        # index dans `texts` ; on remappe en conservant l'ordre.
        reranked: list[dict] = []
        for result in response.results:
            candidate = initial_candidates[result.index]
            candidate["rerank_score"] = result.relevance_score
            reranked.append(candidate)

        return reranked
    
     # ─── Embedding ────────────────────────────────────────────────────────────
    def embed(self, text: str) -> list[float]:
        response = self.embedding_client.embeddings.create(model=self.embed_model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        # On découpe en sous-lots : envoyer des milliers de textes en une seule requête
        # fait saturer (OOM) le runner d'embedding d'Ollama sur GPU.
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = self.embedding_client.embeddings.create(model=self.embed_model, input=batch)
            embeddings.extend([item.embedding for item in response.data])
        return embeddings
    
    def embed_sparse_batch(self, texts: list[str]) -> list[dict]:
        """Retourne une liste de vecteurs creux {indices, values} pour chaque texte."""
        sparse_embeddings = list(self.sparse_model.embed(texts))
        return [
            {"indices": e.indices.tolist(), "values": e.values.tolist()}
            for e in sparse_embeddings
        ]
    
    # ─── Comparaison dense vs hybride ────────────────────────────────────────
    def compare_search(self, prompt: str, top_k: int = 10, output_path: str = "search_comparison.md") -> str:
        """Exécute une recherche dense seule et une recherche hybride pour le même prompt,
        puis écrit les deux résultats (+ un diff) dans un fichier markdown."""
        prompt_dense = self.embed(prompt)
        prompt_sparse = self.embed_sparse_batch([prompt])[0]

        dense_results = self.vector_store.dense_search(prompt_dense, top_k=top_k)
        sparse_results = self.vector_store.sparse_search(prompt_sparse, top_k=top_k)
        hybrid_results = self.vector_store.hybrid_search(prompt_dense, prompt_sparse, top_k=top_k)
        reranked_results = self._rerank_results(prompt, hybrid_results, topk=top_k)

        markdown = self._format_comparison_markdown(prompt, top_k, dense_results, sparse_results, hybrid_results, reranked_results)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return output_path

    def _rerank_results(self, prompt: str, results: list[dict], topk: int) -> list[dict]:
        """Rerange une liste de résultats (dicts avec métadonnées) via le cross-encoder,
        en conservant source/heading et en remplaçant le score par le score de reranking."""
        if not results:
            return []
        # Même logique que rerank_candidates : on donne le heading_path au cross-encoder,
        # sinon les lignes très laconiques (ex. « Pays: Togo, NPF: X... ») perdent leur contexte.
        texts = [f"{r['heading_path']}\n{r['content']}" for r in results]
        response = self.reranker.rerank(
            model=self.reranker_model_name,
            query=prompt,
            documents=texts,
            top_n=topk,
        )
        # Cohere renvoie les résultats déjà triés par pertinence décroissante ;
        # on remplace le score hybride par le score de reranking.
        return [
            {**results[res.index], "score": float(res.relevance_score)}
            for res in response.results
        ]

    def _format_comparison_markdown(self, prompt: str, top_k: int, dense_results: list[dict], sparse_results: list[dict], hybrid_results: list[dict], reranked_results: list[dict]) -> str:
        def escape(text: str) -> str:
            return text.replace("|", "\\|").replace("\n", " ")

        def results_table(results: list[dict]) -> str:
            lines = ["| # | Score | Source | Heading | Content |", "|---|-------|--------|---------|---------|"]
            for i, r in enumerate(results, start=1):
                content = escape(r["content"])[:200]
                lines.append(f"| {i} | {r['score']:.4f} | {r['source']} | {r['heading_path']} | {content} |")
            return "\n".join(lines)

        def reranked_table(results: list[dict], hybrid_results: list[dict], dense_results: list[dict], sparse_results: list[dict]) -> str:
            hybrid_contents = [r["content"] for r in hybrid_results]
            lines = ["| # | Rerank score | Hybrid rank | Dense rank | Sparse rank | Source | Heading | Content |", "|---|--------------|-------------|------------|-------------|--------|---------|---------|"]
            for i, r in enumerate(results, start=1):
                content = escape(r["content"])[:200]
                prev__hyb_rank = hybrid_contents.index(r["content"]) + 1 if r["content"] in hybrid_contents else "—"
                prev__dense_rank = next((j + 1 for j, d in enumerate(dense_results) if d["content"] == r["content"]), "—")
                prev__sparse_rank = next((j + 1 for j, s in enumerate(sparse_results) if s["content"] == r["content"]), "—")
                lines.append(f"| {i} | {r['score']:.4f} | {prev__hyb_rank} | {prev__dense_rank} | {prev__sparse_rank} | {r['source']} | {r['heading_path']} | {content} |")
            return "\n".join(lines)

        sections = [
            f"# Search Comparison\n\n**Prompt:** `{prompt}`\n\n**Top K:** {top_k}\n",
            f"## Dense-only search\n\n{results_table(dense_results)}\n",
            f"## Sparse-only search\n\n{results_table(sparse_results)}\n",
            f"## Hybrid search (dense + BM25, RRF fusion)\n\n{results_table(hybrid_results)}\n",
            f"## Reranked (rerank-v4.0-fast, over hybrid candidates)\n\n_Final candidate list sent to the LLM, hybrid results re-scored by the cross-encoder._\n\n{reranked_table(reranked_results, hybrid_results, dense_results, sparse_results)}\n",
        ]
        return "\n\n".join(sections)