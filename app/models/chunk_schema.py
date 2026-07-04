from pydantic import BaseModel

class ChunkMetadata(BaseModel):
    source: str
    heading_path: str
    chunk_index: int

class DocumentChunk(BaseModel):
    id: str
    vector: list[float]
    sparse_vector: dict
    content: str
    metadata: ChunkMetadata