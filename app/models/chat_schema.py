from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Schema for chat request."""
    prompt: str

class ChatResponse(BaseModel):
    """Schema for chat response."""
    merchandise_name: str
    preferential_rate_applicable: list[str]
