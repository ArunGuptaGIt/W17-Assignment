from pathlib import Path
from typing import Optional, Union
from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import IngestResponse
from app.rag.document_loader import load_document
from app.rag.chunking.recursive import RecursiveChunker
from app.rag.chunking.semantic import SemanticChunker
from app.rag.storage.vector_store import VectorStoreManager
from app.rag.storage.bm25_store import BM25StoreManager
from app.services.cache import RedisCacheManager

class IngestionService:
    """
    Ingestion Service managing document upload, page preservation,
    stable ID creation, versioning, dual indexing, and cache invalidation.
    """

    def __init__(
        self,
        vstore: Optional[VectorStoreManager] = None,
        bm25_store: Optional[BM25StoreManager] = None,
        cache_manager: Optional[RedisCacheManager] = None
    ):
        self.vstore = vstore or VectorStoreManager()
        self.bm25_store = bm25_store or BM25StoreManager()
        self.cache = cache_manager or RedisCacheManager()

        self.recursive_chunker = RecursiveChunker()
        self.semantic_chunker = SemanticChunker()

    async def ingest_file(self, file_path: Union[str, Path]) -> IngestResponse:
        path = Path(file_path)
        filename = path.name

        logger.info(f"Starting ingestion workflow for document: '{filename}'")
        documents = load_document(path)
        if not documents:
            raise ValueError(f"No text content could be extracted from document '{filename}'")

        doc_id = documents[0].metadata.document_id
        total_pages = documents[0].metadata.total_pages

        # Check existing chunk count for this doc_id
        existing_count = self.vstore.client.get_collection("chunks_recursive").get(where={"document_id": doc_id})
        status = "new"

        if existing_count and existing_count.get("ids"):
            logger.info(f"Document '{doc_id}' ({filename}) re-uploaded. Deleting existing chunks for update...")
            self.vstore.delete_document(doc_id)
            self.bm25_store.delete_document(doc_id)
            status = "updated"

        # Generate recursive and semantic chunk variants
        all_r_chunks = []
        all_s_chunks = []

        for doc in documents:
            r_chunks = self.recursive_chunker.chunk_document(doc)
            s_chunks = self.semantic_chunker.chunk_document(doc)
            all_r_chunks.extend(r_chunks)
            all_s_chunks.extend(s_chunks)

        # Dual indexing: Chroma collections
        self.vstore.add_chunks(all_r_chunks, strategy="recursive")
        self.vstore.add_chunks(all_s_chunks, strategy="semantic")

        # Dual indexing: Persistent BM25 indexes
        self.bm25_store.add_chunks(all_r_chunks, strategy="recursive")
        self.bm25_store.add_chunks(all_s_chunks, strategy="semantic")

        # Invalidate stale query cache
        await self.cache.invalidate_all()

        msg = f"Successfully ingested '{filename}'. Created {len(all_r_chunks)} recursive and {len(all_s_chunks)} semantic chunks."
        logger.info(msg)

        return IngestResponse(
            filename=filename,
            document_id=doc_id,
            file_type=documents[0].metadata.file_type,
            total_pages=total_pages,
            recursive_chunks_created=len(all_r_chunks),
            semantic_chunks_created=len(all_s_chunks),
            status=status,
            message=msg
        )
