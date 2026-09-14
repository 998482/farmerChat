from typing import Optional
from pydantic import BaseModel, Field


class KrishiAgentRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Kisan ka sawaal, Hindi ya English mein")
    crop: Optional[str] = Field(None, description="e.g. 'chickpea', 'chana'")
    district: Optional[str] = Field(None, description="e.g. 'sitapur', 'lucknow'")


class KrishiAgentResponse(BaseModel):
    answer: str
    sources: list[str]
    provider_used: str


class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool
