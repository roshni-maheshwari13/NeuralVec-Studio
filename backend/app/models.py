from typing import List, Optional
from pydantic import BaseModel, Field


class VectorInsertRequest(BaseModel):
    vector: Optional[List[float]] = None
    text: Optional[str] = None


class SearchRequest(BaseModel):
    vector: Optional[List[float]] = None
    text: Optional[str] = None
    k: int = Field(default=10, ge=1, le=100)
    method: str = Field(default="ivf")
    nprobe: int = Field(default=5, ge=1)


class SearchResult(BaseModel):
    id: int
    score: float
    text: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    method: str