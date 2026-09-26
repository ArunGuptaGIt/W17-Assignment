# A fixed single-pass pipeline cannot handle complex or ambiguous queries because the model cannot determine in advance whether a single retrieval call returns sufficient context; thus, the number, query formulation, and search mode of retrieval calls must dynamically depend on evaluation of intermediate results.

import time
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import Citation, ChatResponse, RetrievalConfig
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.bm25 import BM25Retriever
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.rerank import CrossEncoderReranker
from app.rag.context_builder import ContextBuilder
from app.services.llm_client import LLMClient

class QAPipeline:
    """
    Agentic RAG QA Pipeline orchestrating Model-Driven Tool Calling Loop
    (search_documents -> Evaluate Context -> Refomulate/Answer/Ask Clarification).
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        context_builder: Optional[ContextBuilder] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.dense = dense_retriever or DenseRetriever()
        self.bm25 = bm25_retriever or BM25Retriever()
        self.hybrid = hybrid_retriever or HybridRetriever(self.dense, self.bm25)
        self.reranker = reranker or CrossEncoderReranker()
        self.context_builder = context_builder or ContextBuilder()
        self.llm_client = llm_client or LLMClient()

    async def answer_question(
        self,
        question: str,
        config: Optional[RetrievalConfig] = None,
        system_prompt_override: Optional[str] = None
    ) -> ChatResponse:
        cfg = config or RetrievalConfig()
        logger.info(
            f"Executing Agentic RAG pipeline [Chunking: '{cfg.chunking}', Mode: '{cfg.mode}', "
            f"Reranker: {cfg.reranker}] for query: '{question}'"
        )

        trace_steps: List[Dict[str, Any]] = []
        all_retrieved_citations: List[Citation] = []
        total_search_calls = 0

        def search_documents_handler(query: str, mode: str = "hybrid") -> Dict[str, Any]:
            nonlocal total_search_calls, all_retrieved_citations
            total_search_calls += 1
            search_mode = mode if mode in ["dense", "bm25", "hybrid"] else cfg.mode

            t0 = time.perf_counter()
            if search_mode == "dense":
                candidates = self.dense.retrieve(query, strategy=cfg.chunking, k=cfg.fusion_top_k)
            elif search_mode == "bm25":
                candidates = self.bm25.retrieve(query, strategy=cfg.chunking, k=cfg.fusion_top_k)
            else:
                candidates = self.hybrid.retrieve(query, strategy=cfg.chunking, fusion_top_k=cfg.fusion_top_k, rrf_k=cfg.rrf_k)
            t_retrieval = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            final_candidates = self.reranker.rerank(query, candidates, top_k=cfg.final_top_k, enabled=cfg.reranker)
            t_rerank = (time.perf_counter() - t0) * 1000

            context_text, citations, is_low_confidence = self.context_builder.build_context(final_candidates)

            for c in citations:
                if not any(existing.chunk_id == c.chunk_id for existing in all_retrieved_citations):
                    all_retrieved_citations.append(c)

            trace_steps.append({
                "name": f"Agent Iteration {total_search_calls}: Document Search",
                "summary": f"Retrieved {len(final_candidates)} chunks via {search_mode.upper()} for query: '{query}'",
                "duration_ms": round(t_retrieval + t_rerank, 2),
                "details": f"Query: '{query}' | Mode: {search_mode} | Low Confidence: {is_low_confidence}"
            })

            return {
                "status": "success",
                "query": query,
                "mode": search_mode,
                "results_count": len(final_candidates),
                "is_low_confidence": is_low_confidence,
                "context": context_text,
                "citations": [c.model_dump() for c in citations]
            }

        sys_prompt = system_prompt_override or (
            "You are an autonomous agentic assistant capable of multi-step research, cross-source verification, and analytical tool execution.\n"
            "You have access to tools: 'search_documents', 'calculator', and 'system_info'.\n\n"
            "Guidelines:\n"
            "1. Multi-Step Research & Search: For multi-part or complex queries, break down the request into sub-queries. Execute search_documents for each topic, evaluate intermediate evidence, and reformulate queries if initial search returns weak context.\n"
            "2. Tool Execution: Use 'calculator' for exact mathematical or numerical computations. Use 'system_info' when hardware metrics are requested.\n"
            "3. Grounding & Cross-Verification: Cross-verify evidence across retrieved chunks. Base factual statements strictly on retrieved context.\n"
            "4. Refusal on Missing Evidence: If retrieved context is insufficient or search returns an error, state clearly: "
            "'I could not find sufficient information in the provided documents to answer that question.' Do NOT fabricate or hallucinate answers."
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question}
        ]

        answer, provider_used, fallback_occurred, latency, usage_stats = await self.llm_client.generate_completion(
            messages=messages,
            enable_tools=True,
            search_handler=search_documents_handler
        )

        trace_steps.append({
            "name": "LLM Agentic Tool Execution Loop",
            "summary": f"Completed agent loop via {provider_used} in {usage_stats.get('iterations', 1)} iteration(s).",
            "duration_ms": round(latency, 2),
            "details": f"Total Tokens: {usage_stats.get('total_tokens', 0)} | Searches Conducted: {total_search_calls}"
        })

        return ChatResponse(
            answer=answer,
            citations=all_retrieved_citations,
            provider_used=provider_used,
            retrieval_config=cfg,
            fallback_occurred=fallback_occurred,
            latency_ms=latency,
            search_results_count=len(all_retrieved_citations),
            trace_steps=trace_steps
        )
