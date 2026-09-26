from typing import Dict, List, Optional
from app.core.config import settings
from app.models.schemas import SearchResult

def reciprocal_rank_fusion(
    dense_results: List[SearchResult],
    bm25_results: List[SearchResult],
    rrf_k: Optional[int] = None,
    top_k: int = 15
) -> List[SearchResult]:
    """
    Combines dense and BM25 search result lists using Reciprocal Rank Fusion.
    RRF score = 1 / (rrf_k + rank_dense) + 1 / (rrf_k + rank_bm25).
    """
    k_const = rrf_k or settings.RRF_K

    rrf_scores: Dict[str, float] = {}
    merged_results: Dict[str, SearchResult] = {}
    dense_scores: Dict[str, float] = {}
    bm25_scores: Dict[str, float] = {}

    # Track dense ranks
    for rank_idx, res in enumerate(dense_results):
        cid = res.chunk_id
        dense_rank = rank_idx + 1
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_const + dense_rank))
        merged_results[cid] = res
        if res.dense_score is not None:
            dense_scores[cid] = res.dense_score

    # Track BM25 ranks
    for rank_idx, res in enumerate(bm25_results):
        cid = res.chunk_id
        bm25_rank = rank_idx + 1
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_const + bm25_rank))
        if cid not in merged_results:
            merged_results[cid] = res
        if res.bm25_score is not None:
            bm25_scores[cid] = res.bm25_score

    # Sort merged results by RRF score descending
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused_results: List[SearchResult] = []
    for final_rank, cid in enumerate(sorted_chunk_ids[:top_k]):
        base_res = merged_results[cid]
        fused = SearchResult(
            document_id=base_res.document_id,
            chunk_id=base_res.chunk_id,
            text=base_res.text,
            metadata=base_res.metadata,
            dense_score=dense_scores.get(cid, base_res.dense_score),
            bm25_score=bm25_scores.get(cid, base_res.bm25_score),
            rrf_score=rrf_scores[cid],
            retrieval_method=f"hybrid_rrf_{base_res.metadata.get('chunking_strategy', 'recursive')}",
            rank=final_rank + 1
        )
        fused_results.append(fused)

    return fused_results
