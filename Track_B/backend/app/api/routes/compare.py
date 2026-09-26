import time
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.models.schemas import CompareRequest, CompareResponse, RetrievalConfig, StrategyComparisonResult
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.bm25 import BM25Retriever
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.rerank import CrossEncoderReranker

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api", tags=["Comparison"])

dense_retriever = DenseRetriever()
bm25_retriever = BM25Retriever()
hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)
reranker = CrossEncoderReranker()

@router.post("/compare", response_model=CompareResponse)
@limiter.limit("2/minute")
async def compare_endpoint(request: Request, body: CompareRequest):
    """
    Developer/Evaluation endpoint running side-by-side comparison of multiple retrieval configurations.
    Executes retrieval & reranking without triggering redundant LLM calls.
    """
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    strategies_to_test = [
        RetrievalConfig(chunking="recursive", mode="hybrid", reranker=True, fusion_top_k=15, final_top_k=body.top_k),
        RetrievalConfig(chunking="semantic", mode="hybrid", reranker=True, fusion_top_k=15, final_top_k=body.top_k),
        RetrievalConfig(chunking="recursive", mode="dense", reranker=True, fusion_top_k=15, final_top_k=body.top_k),
        RetrievalConfig(chunking="semantic", mode="dense", reranker=True, fusion_top_k=15, final_top_k=body.top_k),
    ]

    comparison_results = []

    try:
        for cfg in strategies_to_test:
            start_t = time.perf_counter()

            if cfg.mode == "dense":
                candidates = dense_retriever.retrieve(query, strategy=cfg.chunking, k=cfg.fusion_top_k)
            elif cfg.mode == "bm25":
                candidates = bm25_retriever.retrieve(query, strategy=cfg.chunking, k=cfg.fusion_top_k)
            else:  # hybrid
                candidates = hybrid_retriever.retrieve(query, strategy=cfg.chunking, fusion_top_k=cfg.fusion_top_k)

            final_chunks = reranker.rerank(query, candidates, top_k=cfg.final_top_k, enabled=cfg.reranker)
            latency_ms = (time.perf_counter() - start_t) * 1000.0

            total_tokens = sum(len(c.text) // 4 for c in final_chunks)

            comparison_results.append(StrategyComparisonResult(
                config=cfg,
                retrieved_chunks=final_chunks,
                latency_ms=round(latency_ms, 2),
                total_tokens_used=total_tokens
            ))

        return CompareResponse(query=query, results=comparison_results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison evaluation failed: {str(e)}")
