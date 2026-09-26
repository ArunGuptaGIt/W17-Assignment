from typing import List, Optional
from app.models.schemas import SearchResult
from app.rag.storage.bm25_store import BM25StoreManager

class BM25Retriever:
    """Lexical BM25 retriever querying persistent disk BM25 store."""

    def __init__(self, bm25_store: Optional[BM25StoreManager] = None):
        self.bm25_store = bm25_store or BM25StoreManager()

    def retrieve(self, query: str, strategy: str = "recursive", k: int = 15) -> List[SearchResult]:
        return self.bm25_store.search(query=query, strategy=strategy, k=k)
