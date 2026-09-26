from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    source: str
    title: Optional[str] = None

class Document(BaseModel):
    document_id: str
    text: str
    metadata: DocumentMetadata

class ChunkMetadata(DocumentMetadata):
    chunk_id: str
    chunk_index: int
    chunking_strategy: Literal["recursive", "semantic"]
    character_count: int

class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata

class RetrievalConfig(BaseModel):
    chunking: Literal["recursive", "semantic"] = Field(
        default="recursive",
        description="Chunking strategy to query against ('recursive' or 'semantic')"
    )
    mode: Literal["dense", "bm25", "hybrid"] = Field(
        default="hybrid",
        description="Retrieval search mode ('dense', 'bm25', or 'hybrid')"
    )
    reranker: bool = Field(
        default=True,
        description="Whether cross-encoder reranking is enabled"
    )
    fusion_top_k: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Number of candidate chunks to retrieve during dense/BM25/RRF stage"
    )
    final_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top chunks to include in final LLM context"
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Reciprocal Rank Fusion smoothing constant"
    )

    @model_validator(mode="after")
    def validate_top_k(self) -> "RetrievalConfig":
        if self.final_top_k > self.fusion_top_k:
            raise ValueError(
                f"final_top_k ({self.final_top_k}) cannot be larger than fusion_top_k ({self.fusion_top_k})."
            )
        return self

class SearchResult(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    reranker_score: Optional[float] = None
    retrieval_method: str
    rank: Optional[int] = None
    final_rank: Optional[int] = None

class ChatRequest(BaseModel):
    message: str
    config: Optional[RetrievalConfig] = Field(default_factory=RetrievalConfig)
    stream: bool = False
    system_prompt_override: Optional[str] = None

class Citation(BaseModel):
    filename: str
    page_number: Optional[int] = None
    chunk_id: str
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    provider_used: str
    retrieval_config: RetrievalConfig
    fallback_occurred: bool = False
    latency_ms: float
    search_results_count: int
    trace_steps: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

class IngestResponse(BaseModel):
    filename: str
    document_id: str
    file_type: str
    total_pages: Optional[int] = None
    recursive_chunks_created: int
    semantic_chunks_created: int
    status: Literal["new", "updated", "unchanged"]
    message: str

class CompareRequest(BaseModel):
    query: str
    top_k: int = 5

class StrategyComparisonResult(BaseModel):
    config: RetrievalConfig
    retrieved_chunks: List[SearchResult]
    latency_ms: float
    total_tokens_used: int

class CompareResponse(BaseModel):
    query: str
    results: List[StrategyComparisonResult]
