from typing import List, Tuple
from app.core.config import settings
from app.models.schemas import Citation, SearchResult

class ContextBuilder:
    """
    Context Builder stage:
    Deduplicates chunks, evaluates retrieval confidence threshold ("I don't know"),
    applies token budget limits, and formats source citations for LLM context.
    """

    def __init__(
        self,
        max_context_tokens: int = settings.MAX_CONTEXT_TOKENS,
        confidence_threshold: float = settings.CONFIDENCE_THRESHOLD,
        idk_response: str = settings.IDK_RESPONSE
    ):
        self.max_context_tokens = max_context_tokens
        self.confidence_threshold = confidence_threshold
        self.idk_response = idk_response

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimation (approx 4 chars per token)."""
        return max(1, len(text) // 4)

    def build_context(self, search_results: List[SearchResult]) -> Tuple[str, List[Citation], bool]:
        """
        Build formatted LLM context string and citations list.
        Surfaces retrieval/reranking scores directly in context headers for model evaluation.
        Returns (context_text, citations_list, is_low_confidence).
        """
        if not search_results:
            return "No relevant document chunks found.", [], True

        # Calculate max score for informational telemetry
        max_score = 0.0
        for res in search_results:
            score = (
                res.reranker_score if res.reranker_score is not None else
                res.rrf_score if res.rrf_score is not None else
                res.dense_score if res.dense_score is not None else
                res.bm25_score if res.bm25_score is not None else 0.0
            )
            max_score = max(max_score, score)

        # Deduplicate chunks by chunk_id
        seen_chunk_ids = set()
        deduped_results: List[SearchResult] = []
        for res in search_results:
            if res.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(res.chunk_id)
                deduped_results.append(res)

        context_blocks: List[str] = []
        citations: List[Citation] = []
        current_token_count = 0

        for res in deduped_results:
            filename = res.metadata.get("filename", "unknown")
            page_num = res.metadata.get("page_number")
            page_str = f", Page: {page_num}" if page_num else ""
            
            score_str = ""
            if res.reranker_score is not None:
                score_str = f" [Reranker Score: {res.reranker_score:.3f}]"
            elif res.rrf_score is not None:
                score_str = f" [RRF Score: {res.rrf_score:.3f}]"
            elif res.dense_score is not None:
                score_str = f" [Dense Score: {res.dense_score:.3f}]"
            elif res.bm25_score is not None:
                score_str = f" [BM25 Score: {res.bm25_score:.3f}]"

            header = f"--- Document: {filename}{page_str}{score_str} [Chunk ID: {res.chunk_id}] ---"
            block = f"{header}\n{res.text}\n"

            block_tokens = self._estimate_tokens(block)
            if current_token_count + block_tokens > self.max_context_tokens:
                break

            context_blocks.append(block)
            current_token_count += block_tokens

            citations.append(Citation(
                filename=filename,
                page_number=page_num,
                chunk_id=res.chunk_id,
                snippet=res.text[:150] + ("..." if len(res.text) > 150 else "")
            ))

        if not context_blocks:
            return "No relevant document chunks found.", [], True

        formatted_context = "\n".join(context_blocks)
        is_low_confidence = max_score < self.confidence_threshold
        return formatted_context, citations, is_low_confidence
