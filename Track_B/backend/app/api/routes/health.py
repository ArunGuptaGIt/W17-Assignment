from fastapi import APIRouter
from app.core.config import settings
from app.rag.storage.vector_store import VectorStoreManager
from app.rag.storage.bm25_store import BM25StoreManager
from app.services.cache import RedisCacheManager

router = APIRouter(prefix="/api", tags=["Health"])

vstore = VectorStoreManager()
bm25_store = BM25StoreManager()
cache_manager = RedisCacheManager()

@router.get("/health")
async def health_check():
    """Liveness probe indicating API process is running."""
    return {"status": "healthy", "service": settings.PROJECT_NAME}

@router.get("/ready")
async def readiness_check():
    """
    Readiness probe testing local storage dependencies (ChromaDB, BM25 Store, Redis).
    Does NOT fail if external Gemini provider is offline due to local/NIM fallbacks.
    """
    readiness = {
        "chroma_db": "unknown",
        "bm25_store": "unknown",
        "redis_cache": "unknown"
    }

    # Check ChromaDB
    try:
        r_count = vstore.get_chunk_count("recursive")
        s_count = vstore.get_chunk_count("semantic")
        readiness["chroma_db"] = f"ready (chunks_recursive: {r_count}, chunks_semantic: {s_count})"
    except Exception as e:
        readiness["chroma_db"] = f"error: {str(e)}"

    # Check BM25 Store
    try:
        bm25_r = bm25_store.get_chunk_count("recursive")
        bm25_s = bm25_store.get_chunk_count("semantic")
        readiness["bm25_store"] = f"ready (recursive: {bm25_r}, semantic: {bm25_s})"
    except Exception as e:
        readiness["bm25_store"] = f"error: {str(e)}"

    # Check Redis Cache
    try:
        client = await cache_manager.get_client()
        if client:
            readiness["redis_cache"] = "ready"
        else:
            readiness["redis_cache"] = "disabled/offline (graceful fallback)"
    except Exception as e:
        readiness["redis_cache"] = f"error: {str(e)}"

    is_ready = "error" not in str(readiness.values())

    return {
        "status": "ready" if is_ready else "degraded",
        "components": readiness
    }
