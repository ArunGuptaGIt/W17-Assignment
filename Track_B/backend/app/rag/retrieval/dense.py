from typing import List, Optional
from app.models.schemas import SearchResult
from app.rag.storage.vector_store import VectorStoreManager

class DenseRetriever:
    """Dense vector retriever querying ChromaDB vector store."""

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()

    def retrieve(self, query: str, strategy: str = "recursive", k: int = 15) -> List[SearchResult]:
        return self.vector_store.similarity_search(query=query, strategy=strategy, k=k)
