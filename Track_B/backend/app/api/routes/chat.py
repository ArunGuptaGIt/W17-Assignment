from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.models.schemas import ChatRequest, ChatResponse
from app.services.cache import RedisCacheManager
from app.services.qa_pipeline import QAPipeline

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api", tags=["Chat"])

qa_pipeline = QAPipeline()
cache_manager = RedisCacheManager()

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat_endpoint(request: Request, body: ChatRequest):
    """
    Primary chat/completion endpoint executing ONE validated RetrievalConfig pipeline.
    Strategy-aware Redis caching checked before calling LLM.
    """
    try:
        config = body.config
        query = body.message.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query message cannot be empty.")

        # Check Redis Cache
        cache_key = cache_manager.generate_cache_key(query=query, config=config)
        cached_res = await cache_manager.get(cache_key)
        if cached_res:
            return ChatResponse(**cached_res)

        # Run RAG QA Pipeline
        response = await qa_pipeline.answer_question(
            question=query,
            config=config,
            system_prompt_override=body.system_prompt_override
        )

        # Save to Redis Cache
        await cache_manager.set(cache_key, response.model_dump())

        return response

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal RAG pipeline error: {str(e)}")
