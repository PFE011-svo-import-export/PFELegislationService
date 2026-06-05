from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Schema for chat request."""
    prompt: str

class ChatResponse(BaseModel):
    """Schema for chat response."""
    response: str
