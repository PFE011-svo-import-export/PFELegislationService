from ollama import Client as OllamaClient
from mistletoe import Document
from mistletoe.ast_renderer import ASTRenderer
from app.storage.qdrant_vectordb import VectorStore
import json
import mistletoe
import uuid
import os
from app.models.chunk_schema import ChunkMetadata, DocumentChunk

class RagService:
    def __init__(self, client: OllamaClient, embed_model: str, vector_store: VectorStore):
        self.client = client
        self.embed_model = embed_model
        self.vector_store = vector_store

     # ─── Embedding ────────────────────────────────────────────────────────────
    def embed(self, text: str) -> list[float]:
        response = self.client.embed(model=self.embed_model, input=text)
        return response.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(model=self.embed_model, input=texts)
        return response.embeddings
    
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

        Retourne:
            [
                {
                    "content": "...",
                    "metadata": {
                        "source": "fichier.md",
                        "heading_path": "H1 > H2",
                        "chunk_index": 0
                    }
                },
                ...
            ]
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
        vectors = self.embed_batch(texts)
        
        data = [
            {
                "vector": vector,
                "content": chunk["content"],
                "metadata": chunk["metadata"],
            }
            for vector, chunk in zip(vectors, chunks)
        ]
        
        chunks_to_store = [
            DocumentChunk(
                id=str(uuid.uuid4()),
                vector=d["vector"],
                content=d["content"],
                metadata=ChunkMetadata(**d["metadata"])
            )
            for d in data
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

    def delete_coll(self):
        self.vector_store.delete_collection()