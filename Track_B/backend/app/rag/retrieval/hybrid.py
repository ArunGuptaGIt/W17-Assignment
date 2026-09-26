from typing import List, Optional
from app.core.config import settings
from app.models.schemas import SearchResult
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.bm25 import BM25Retriever
from app.rag.retrieval.rrf import reciprocal_rank_fusion

class HybridRetriever:
    """
    Hybrid retriever orchestrating Dense and BM25 retrievers
    and fusing candidate lists using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None
    ):
        self.dense = dense_retriever or DenseRetriever()
        self.bm25 = bm25_retriever or BM25Retriever()

    def retrieve(
        self,
        query: str,
        strategy: str = "recursive",
        fusion_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None
    ) -> List[SearchResult]:
        k_val = fusion_top_k or settings.FUSION_TOP_K
        k_rrf = rrf_k or settings.RRF_K

        dense_results = self.dense.retrieve(query=query, strategy=strategy, k=k_val)
        bm25_results = self.bm25.retrieve(query=query, strategy=strategy, k=k_val)

        return reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            rrf_k=k_rrf,
            top_k=k_val
        )
