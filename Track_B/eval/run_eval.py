import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.core.config import settings
from app.models.schemas import RetrievalConfig, SearchResult
from app.rag.document_loader import load_document
from app.rag.chunking.recursive import RecursiveChunker
from app.rag.chunking.semantic import SemanticChunker
from app.rag.storage.vector_store import VectorStoreManager
from app.rag.storage.bm25_store import BM25StoreManager
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.bm25 import BM25Retriever
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.rerank import CrossEncoderReranker

def ingest_eval_documents(vstore: VectorStoreManager, bm25_store: BM25StoreManager):
    """Index sample documents into Chroma and BM25 store for evaluation."""
    doc_files = list(settings.DOCUMENTS_DIR.glob("*.*"))
    if not doc_files:
        print("No documents found in settings.DOCUMENTS_DIR.")
        return

    r_chunker = RecursiveChunker()
    s_chunker = SemanticChunker()

    for doc_path in doc_files:
        docs = load_document(doc_path)
        for doc in docs:
            r_chunks = r_chunker.chunk_document(doc)
            s_chunks = s_chunker.chunk_document(doc)

            vstore.add_chunks(r_chunks, strategy="recursive")
            vstore.add_chunks(s_chunks, strategy="semantic")

            bm25_store.add_chunks(r_chunks, strategy="recursive")
            bm25_store.add_chunks(s_chunks, strategy="semantic")

def is_chunk_relevant(chunk: SearchResult, item: Dict[str, Any]) -> bool:
    """Check if retrieved chunk matches ground truth document or contains ground truth keywords."""
    filename = chunk.metadata.get("filename", "")
    text_lower = chunk.text.lower()
    
    doc_match = any(doc_name in filename for doc_name in item["relevant_documents"])
    keyword_matches = sum(1 for kw in item["keywords"] if kw.lower() in text_lower)

    return doc_match and (keyword_matches > 0)

def calculate_metrics(results: List[SearchResult], item: Dict[str, Any], k: int) -> Tuple[float, float, float, float]:
    """Calculate Precision@K, Recall@K, MRR, and nDCG@K for a query result."""
    top_k_results = results[:k]
    if not top_k_results:
        return 0.0, 0.0, 0.0, 0.0

    relevances = [1.0 if is_chunk_relevant(res, item) else 0.0 for res in top_k_results]
    
    # Precision@K
    precision = sum(relevances) / k

    # Recall@K (estimated against total keywords)
    total_keywords = len(item["keywords"])
    found_keywords = set()
    for res in top_k_results:
        text_lower = res.text.lower()
        for kw in item["keywords"]:
            if kw.lower() in text_lower:
                found_keywords.add(kw)
    recall = len(found_keywords) / max(1, total_keywords)

    # MRR
    mrr = 0.0
    for rank_idx, rel in enumerate(relevances):
        if rel > 0:
            mrr = 1.0 / (rank_idx + 1)
            break

    # nDCG@K
    dcg = sum(rel / math.log2(rank_idx + 2) for rank_idx, rel in enumerate(relevances))
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(rank_idx + 2) for rank_idx, rel in enumerate(ideal_relevances))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return precision, recall, mrr, ndcg

def run_evaluation():
    vstore = VectorStoreManager()
    bm25_store = BM25StoreManager()

    print("Ingesting evaluation documents...")
    ingest_eval_documents(vstore, bm25_store)

    dense_retriever = DenseRetriever(vstore)
    bm25_retriever = BM25Retriever(bm25_store)
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)
    reranker = CrossEncoderReranker()

    dataset_path = Path(__file__).parent / "datasets" / "ground_truth.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        gt_dataset = json.load(f)

    chunking_options = ["recursive", "semantic"]
    retrieval_options = ["dense", "bm25", "hybrid"]
    reranker_options = [False, True]

    eval_results = []

    print(f"\n--- Running 12-Matrix Retrieval Evaluation Across {len(gt_dataset)} Test Queries ---\n")

    for chunking in chunking_options:
        for mode in retrieval_options:
            for use_reranker in reranker_options:
                config = RetrievalConfig(
                    chunking=chunking,
                    mode=mode,
                    reranker=use_reranker,
                    fusion_top_k=10,
                    final_top_k=5
                )

                precisions, recalls, mrrs, ndcgs = [], [], [], []
                total_latencies = []

                for item in gt_dataset:
                    query = item["question"]
                    start_t = time.perf_counter()

                    # Retrieval stage
                    if mode == "dense":
                        candidates = dense_retriever.retrieve(query, strategy=chunking, k=config.fusion_top_k)
                    elif mode == "bm25":
                        candidates = bm25_retriever.retrieve(query, strategy=chunking, k=config.fusion_top_k)
                    else:  # hybrid
                        candidates = hybrid_retriever.retrieve(query, strategy=chunking, fusion_top_k=config.fusion_top_k)

                    # Rerank stage
                    final_chunks = reranker.rerank(query, candidates, top_k=config.final_top_k, enabled=use_reranker)
                    latency = (time.perf_counter() - start_t) * 1000.0

                    p, r, m, n = calculate_metrics(final_chunks, item, k=config.final_top_k)
                    precisions.append(p)
                    recalls.append(r)
                    mrrs.append(m)
                    ndcgs.append(n)
                    total_latencies.append(latency)

                avg_p = sum(precisions) / len(precisions)
                avg_r = sum(recalls) / len(recalls)
                avg_m = sum(mrrs) / len(mrrs)
                avg_n = sum(ndcgs) / len(ndcgs)
                avg_lat = sum(total_latencies) / len(total_latencies)

                eval_results.append({
                    "chunking": chunking,
                    "mode": mode,
                    "reranker": "ON" if use_reranker else "OFF",
                    "precision_at_5": avg_p,
                    "recall_at_5": avg_r,
                    "mrr": avg_m,
                    "ndcg_at_5": avg_n,
                    "latency_ms": avg_lat
                })

    # Print summary markdown table
    print("| Chunking | Mode | Reranker | Precision@5 | Recall@5 | MRR | nDCG@5 | Latency (ms) |")
    print("|---|---|---|---|---|---|---|---|")
    for res in eval_results:
        print(f"| {res['chunking']} | {res['mode']} | {res['reranker']} | {res['precision_at_5']:.3f} | {res['recall_at_5']:.3f} | {res['mrr']:.3f} | {res['ndcg_at_5']:.3f} | {res['latency_ms']:.1f} |")

if __name__ == "__main__":
    run_evaluation()
