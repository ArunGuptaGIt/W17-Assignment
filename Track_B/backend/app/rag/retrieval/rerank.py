from typing import List, Optional
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import SearchResult

class CrossEncoderReranker:
    """
    Cross-Encoder Reranker scoring candidate search results.
    Configured for CPU/GPU execution to avoid VRAM contention on 4GB GPUs.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self.device = device or settings.RERANKER_DEVICE
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            logger.info(f"Loading CrossEncoder reranker model: '{self.model_name}' on device: '{self.device}'")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5,
        enabled: bool = True
    ) -> List[SearchResult]:
        if not candidates or not enabled:
            return candidates[:top_k]

        pairs = [(query, cand.text) for cand in candidates]
        logger.info(f"Reranking {len(candidates)} candidate chunks using {self.model_name}...")
        
        scores = self.model.predict(pairs, show_progress_bar=False)

        reranked_candidates = []
        for cand, score in zip(candidates, scores):
            # Clone candidate and attach score
            cloned = cand.model_copy(update={
                "reranker_score": float(score),
                "retrieval_method": f"{cand.retrieval_method}_reranked"
            })
            reranked_candidates.append(cloned)

        # Sort descending by reranker_score
        reranked_candidates.sort(key=lambda x: x.reranker_score or -999.0, reverse=True)

        final_list = []
        for idx, res in enumerate(reranked_candidates[:top_k]):
            res.final_rank = idx + 1
            final_list.append(res)

        return final_list
