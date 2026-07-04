from ollama import Client as OllamaClient
from mistletoe import Document
from mistletoe.ast_renderer import ASTRenderer
from app.storage.qdrant_vectordb import VectorStore
import json
import mistletoe
import uuid
import os
from app.models.chunk_schema import ChunkMetadata, DocumentChunk
from FlagEmbedding import FlagReranker
from fastembed import SparseTextEmbedding

class RagService:
    def __init__(self, client: OllamaClient, embed_model: str, vector_store: VectorStore, reranker_model: FlagReranker):
        self.client = client
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.reranker = reranker_model
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

     # ─── Embedding ────────────────────────────────────────────────────────────
    def embed(self, text: str) -> list[float]:
        response = self.client.embed(model=self.embed_model, input=text)
        return response.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(model=self.embed_model, input=texts)
        return response.embeddings
    
    def embed_sparse_batch(self, texts: list[str]) -> list[dict]:
        """Retourne une liste de vecteurs creux {indices, values} pour chaque texte."""
        sparse_embeddings = list(self.sparse_model.embed(texts))
        return [
            {"indices": e.indices.tolist(), "values": e.values.tolist()}
            for e in sparse_embeddings
        ]
    
    # ─── Parsing ──────────────────────────────────────────────────────────────
    def parse_md_file(self, filepath: str) -> dict:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        with ASTRenderer():
            Document(content)
            return json.loads(mistletoe.markdown(content, ASTRenderer))
        
    # ─── Extraction de texte ──────────────────────────────────────────────────
    def _extract_text(self, node: dict) -> str:
        """Extrait le texte brut récursivement, sans formatage inline."""
        t = node.get("type")
        if t == "RawText":
            return node.get("content", "")
        if t == "InlineCode":
            return "".join(self._extract_text(c) for c in node.get("children", []) or [])
        return " ".join(
            self._extract_text(c) for c in node.get("children", []) or []
        ).strip()
    
    # ─── Sérialiseurs ─────────────────────────────────────────────────────────
    def _serialize_paragraph(self, node: dict) -> str:
        return self._extract_text(node)

    def _serialize_list(self, node: dict) -> str:
        start = node.get("start")
        is_ordered = start is not None
        lines = []
        for i, item in enumerate(node.get("children", []), start=start if is_ordered else 1):
            prefix = f"{i}." if is_ordered else "-"
            lines.append(f"{prefix} {self._extract_text(item)}")
        return "\n".join(lines)

    def _serialize_quote(self, node: dict) -> str:
        parts = [self._extract_text(child) for child in node.get("children", []) or []]
        return "[Note] " + " ".join(parts).strip()

    def _serialize_table(self, node: dict) -> str:
        headers = [self._extract_text(cell) for cell in node["header"].get("children", [])]
        lines = []
        for row in node.get("children", []):
            cells = [self._extract_text(cell) for cell in row.get("children", [])]
            lines.append(", ".join(f"{h}: {v}" for h, v in zip(headers, cells)))
        return "\n".join(lines)

    def _serialize_code_fence(self, node: dict) -> str:
        lang = node.get("language", "")
        code = (node.get("children", [{}]) or [{}])[0].get("content", "")
        return f"```{lang}\n{code}\n```"

    def _serialize_node(self, node: dict) -> str | None:
        """Convertit un nœud en texte. Retourne None si ignoré."""
        handlers = {
            "Paragraph": self._serialize_paragraph,
            "List":      self._serialize_list,
            "Quote":     self._serialize_quote,
            "Table":     self._serialize_table,
            "CodeFence": self._serialize_code_fence,
        }
        handler = handlers.get(node.get("type"))
        return handler(node) if handler else None

    # ─── Chunking ─────────────────────────────────────────────────────────────
    def chunk_md_file(self, filepath: str) -> list[dict]:
        """
        Parse un fichier .md et retourne une liste de chunks avec métadonnées.
        Chaque chunk correspond à une section délimitée par un heading.

        """
        print(f"Parsing and chunking file: {filepath}")
        ast = self.parse_md_file(filepath)
        source = os.path.basename(filepath)

        chunks = []
        # Contexte de heading courant par niveau (1..6)
        heading_stack = {}
        current_content_lines = []
        chunk_index = 0

        def flush_chunk():
            """Sauvegarde le contenu accumulé comme chunk."""
            nonlocal chunk_index
            content = "\n\n".join(current_content_lines).strip()
            if not content:
                return
            # Construire le chemin hiérarchique ex: "Introduction > Authentification"
            heading_path = " > ".join(
                heading_stack[lvl]
                for lvl in sorted(heading_stack)
            )
            chunks.append({
                "content": content,
                "metadata": {
                    "source": source,
                    "heading_path": heading_path,
                    "chunk_index": chunk_index,
                }
            })
            chunk_index += 1
            current_content_lines.clear()

        for node in ast.get("children", []):
            t = node.get("type")

            if t == "Heading":
                # Nouveau heading = nouvelle section = flush du chunk précédent
                flush_chunk()
                level = node.get("level", 1)
                title = self._extract_text(node)
                # Mettre à jour le stack et effacer les niveaux enfants
                heading_stack[level] = title
                heading_stack = {k: v for k, v in heading_stack.items() if k <= level}
            else:
                serialized = self._serialize_node(node)
                if serialized:
                    current_content_lines.append(serialized)

        flush_chunk()  # dernière section
        return chunks

    # ─── Pipeline complet ─────────────────────────────────────────────────────

    def embed_md_file(self, filepath: str) -> list[dict]:
        chunks = self.chunk_md_file(filepath)
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
    

    # ─── Comparaison dense vs hybride ────────────────────────────────────────
    def compare_search(self, prompt: str, top_k: int = 10, output_path: str = "search_comparison.md") -> str:
        """Exécute une recherche dense seule et une recherche hybride pour le même prompt,
        puis écrit les deux résultats (+ un diff) dans un fichier markdown."""
        prompt_dense = self.embed(prompt)
        prompt_sparse = self.embed_sparse_batch([prompt])[0]

        dense_results = self.vector_store.search(prompt_dense, top_k=top_k)
        hybrid_results = self.vector_store.hybrid_search(prompt_dense, prompt_sparse, top_k=top_k)
        reranked_results = self._rerank_results(prompt, hybrid_results, topk=top_k)

        markdown = self._format_comparison_markdown(prompt, top_k, dense_results, hybrid_results, reranked_results)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return output_path

    def _rerank_results(self, prompt: str, results: list[dict], topk: int) -> list[dict]:
        """Rerange une liste de résultats (dicts avec métadonnées) via le cross-encoder,
        en conservant source/heading et en remplaçant le score par le score de reranking."""
        if not results:
            return []
        pairs = [[prompt, r["content"]] for r in results]
        scores = self.reranker.compute_score(pairs, normalize=True, batch_size=16)
        reranked = sorted(
            ({**r, "score": float(s)} for r, s in zip(results, scores)),
            key=lambda r: r["score"],
            reverse=True,
        )
        return reranked[:topk]

    def _format_comparison_markdown(self, prompt: str, top_k: int, dense_results: list[dict], hybrid_results: list[dict], reranked_results: list[dict]) -> str:
        def escape(text: str) -> str:
            return text.replace("|", "\\|").replace("\n", " ")

        def results_table(results: list[dict]) -> str:
            lines = ["| # | Score | Source | Heading | Content |", "|---|-------|--------|---------|---------|"]
            for i, r in enumerate(results, start=1):
                content = escape(r["content"])[:200]
                lines.append(f"| {i} | {r['score']:.4f} | {r['source']} | {r['heading_path']} | {content} |")
            return "\n".join(lines)

        def reranked_table(results: list[dict], hybrid_results: list[dict]) -> str:
            hybrid_contents = [r["content"] for r in hybrid_results]
            lines = ["| # | Rerank score | Hybrid rank | Source | Heading | Content |", "|---|--------------|-------------|--------|---------|---------|"]
            for i, r in enumerate(results, start=1):
                content = escape(r["content"])[:200]
                prev_rank = hybrid_contents.index(r["content"]) + 1 if r["content"] in hybrid_contents else "—"
                lines.append(f"| {i} | {r['score']:.4f} | {prev_rank} | {r['source']} | {r['heading_path']} | {content} |")
            return "\n".join(lines)

        dense_contents = [r["content"] for r in dense_results]
        hybrid_contents = [r["content"] for r in hybrid_results]

        only_dense = [c for c in dense_contents if c not in hybrid_contents]
        only_hybrid = [c for c in hybrid_contents if c not in dense_contents]
        common = [c for c in dense_contents if c in hybrid_contents]

        rank_lines = ["| Content | Dense rank | Hybrid rank |", "|---------|------------|-------------|"]
        for c in common:
            rank_lines.append(f"| {escape(c)[:150]} | {dense_contents.index(c) + 1} | {hybrid_contents.index(c) + 1} |")

        sections = [
            f"# Search Comparison\n\n**Prompt:** `{prompt}`\n\n**Top K:** {top_k}\n",
            f"## Dense-only search\n\n{results_table(dense_results)}\n",
            f"## Hybrid search (dense + BM25, RRF fusion)\n\n{results_table(hybrid_results)}\n",
            f"## Reranked (bge-reranker-v2-m3, over hybrid candidates)\n\n_Final candidate list sent to the LLM, hybrid results re-scored by the cross-encoder._\n\n{reranked_table(reranked_results, hybrid_results)}\n",
            f"## Diff\n\n**Only in dense ({len(only_dense)}):**\n\n" + "\n".join(f"- {escape(c)[:150]}" for c in only_dense) if only_dense else "**Only in dense (0):** none",
            f"**Only in hybrid ({len(only_hybrid)}):**\n\n" + "\n".join(f"- {escape(c)[:150]}" for c in only_hybrid) if only_hybrid else "**Only in hybrid (0):** none",
            f"**Common results ({len(common)}), rank comparison:**\n\n{chr(10).join(rank_lines) if common else '_none_'}",
        ]
        return "\n\n".join(sections)

    def retrieve(self, prompt: str) -> list[str]:
        prompt_dense = self.embed(prompt)
        prompt_sparse = self.embed_sparse_batch([prompt])[0]

        print("Retrieving the candidates from db...")
        initial_candidates = self.vector_store.hybrid_search(prompt_dense, prompt_sparse)
        #initial_candidates = self.vector_store.search(prompt_dense, 15)

        print(f"******************* Initial candidates ********************** \n")

        for c in initial_candidates:
            print(f"[{c["score"]}]  {c["content"]} \n")
        
        content = [item["content"] for item in initial_candidates]

        return self.rerank_candidates(prompt, content, 5)

    def rerank_candidates(self, prompt: str, initial_candidates: list[str], topk: int) -> list[str]:
        pairs = [[prompt, candidate] for candidate in initial_candidates]

        scores = self.reranker.compute_score(pairs, normalize=True, batch_size=16)

        # On trie les candidats par score décroissant
        reranked = sorted(zip(scores, initial_candidates), key=lambda x: x[0], reverse=True)

        print(f"******************* Reranked candidates ********************** \n")

        for score, candidate in reranked:
            print(f"{score:.4f} — {candidate}")
        
        top_candidates = [candidate for score, candidate in reranked[:topk]]
        return top_candidates

    def delete_coll(self):
        self.vector_store.delete_collection()