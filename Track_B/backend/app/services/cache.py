import hashlib
import json
from typing import Any, Dict, Optional
import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import RetrievalConfig

class RedisCacheManager:
    """
    Strategy-Aware Redis Query Cache Manager.
    Generates deterministic cache keys containing query string, chunking strategy,
    retrieval mode, reranker toggle, and top_k parameters.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self.enabled = settings.CACHE_ENABLED
        self.ttl = settings.CACHE_TTL_SECONDS
        self._client: Optional[redis.Redis] = None

    async def get_client(self) -> Optional[redis.Redis]:
        if not self.enabled:
            return None
        if self._client is None:
            try:
                self._client = redis.from_url(self.redis_url, decode_responses=True)
                await self._client.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed ({e}). Running with cache disabled.")
                self._client = None
        return self._client

    def generate_cache_key(self, query: str, config: RetrievalConfig, index_version: str = "v1") -> str:
        """
        Generate strategy-aware cache key:
        hash(query + chunking + mode + reranker + fusion_k + final_k + index_version)
        """
        raw_key = (
            f"{query.strip().lower()}|"
            f"{config.chunking}|"
            f"{config.mode}|"
            f"{config.reranker}|"
            f"{config.fusion_top_k}|"
            f"{config.final_top_k}|"
            f"{index_version}"
        )
        hashed = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return f"rag:cache:{hashed}"

    async def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        client = await self.get_client()
        if not client:
            return None
        try:
            cached_data = await client.get(cache_key)
            if cached_data:
                logger.info(f"Cache HIT for key: {cache_key[:20]}...")
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Failed to read from Redis cache: {e}")
        return None

    async def set(self, cache_key: str, data: Dict[str, Any], ttl: Optional[int] = None):
        client = await self.get_client()
        if not client:
            return
        try:
            serialized = json.dumps(data)
            await client.set(cache_key, serialized, ex=ttl or self.ttl)
            logger.info(f"Cache SET for key: {cache_key[:20]}...")
        except Exception as e:
            logger.warning(f"Failed to write to Redis cache: {e}")

    async def invalidate_all(self):
        client = await self.get_client()
        if not client:
            return
        try:
            keys = await client.keys("rag:cache:*")
            if keys:
                await client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} RAG cache entries.")
        except Exception as e:
            logger.warning(f"Failed to invalidate Redis cache: {e}")
