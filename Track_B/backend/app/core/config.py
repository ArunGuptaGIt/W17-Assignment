from pathlib import Path
from typing import Literal
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root .env path (works regardless of working directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project metadata
    PROJECT_NAME: str = "Enterprise RAG Assistant"
    DEBUG: bool = False

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    INDEXES_DIR: Path = DATA_DIR / "indexes"
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma_db"

    # Strategy defaults
    DEFAULT_CHUNKING_STRATEGY: Literal["recursive", "semantic"] = "recursive"
    DEFAULT_RETRIEVAL_MODE: Literal["dense", "bm25", "hybrid"] = "hybrid"
    ENABLE_RERANKER: bool = True

    # Retrieval Top-K & Fusion settings
    FUSION_TOP_K: int = 15
    FINAL_TOP_K: int = 5
    RRF_K: int = 60

    # Chunking parameters
    RECURSIVE_CHUNK_SIZE: int = 500
    RECURSIVE_CHUNK_OVERLAP: int = 50

    SEMANTIC_BREAKPOINT_PERCENTILE: float = 95.0
    SEMANTIC_BUFFER_SIZE: int = 1
    SEMANTIC_MIN_CHUNK_SIZE: int = 100
    SEMANTIC_MAX_CHUNK_SIZE: int = 1500

    # Models & Devices
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BACKEND: Literal["pytorch", "onnx_fp32", "onnx_int8"] = "pytorch"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_DEVICE: Literal["cpu", "cuda", "auto"] = "cpu"

    # Redis Cache Settings
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600
    CACHE_ENABLED: bool = True

    # API Keys & Endpoints
    GEMINI_API_KEY: str = ""
    VLLM_BASE_URL: str = "http://localhost:8001/v1"
    VLLM_TIMEOUT: float = 60.0

    # LLM Models
    PRIMARY_MODEL: str = "gemini-1.5-flash-latest"
    LOCAL_VLLM_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"

    # Context Builder & Confidence limits
    MAX_CONTEXT_TOKENS: int = 2048
    MAX_OUTPUT_TOKENS: int = 1024
    MAX_AGENT_ITERATIONS: int = 4
    TEMPERATURE: float = 0.2
    TOP_P: float = 0.95
    CONFIDENCE_THRESHOLD: float = 0.05
    IDK_RESPONSE: str = "I couldn't find enough relevant information in the provided documents to answer that question."

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        if self.RECURSIVE_CHUNK_OVERLAP >= self.RECURSIVE_CHUNK_SIZE:
            raise ValueError(
                f"RECURSIVE_CHUNK_OVERLAP ({self.RECURSIVE_CHUNK_OVERLAP}) must be strictly less than "
                f"RECURSIVE_CHUNK_SIZE ({self.RECURSIVE_CHUNK_SIZE})."
            )
        if self.SEMANTIC_MIN_CHUNK_SIZE >= self.SEMANTIC_MAX_CHUNK_SIZE:
            raise ValueError(
                f"SEMANTIC_MIN_CHUNK_SIZE ({self.SEMANTIC_MIN_CHUNK_SIZE}) must be strictly less than "
                f"SEMANTIC_MAX_CHUNK_SIZE ({self.SEMANTIC_MAX_CHUNK_SIZE})."
            )
        return self

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
settings.INDEXES_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
