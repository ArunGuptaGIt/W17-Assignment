import pickle
import re

from pathlib import Path
from typing import Dict, List, Optional
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import Chunk, SearchResult

def tokenize_text(text: str) -> List[str]:
    """Lowercase text and extract alphanumeric word tokens."""
    return re.findall(r'\w+', text.lower())

class BM25IndexData:
    """Serializable container storing BM25Okapi model, raw chunks, and tokenized corpus."""
    def __init__(self, bm25: BM25Okapi, chunks: List[Chunk], corpus_tokens: List[List[str]]):
        self.bm25 = bm25
        self.chunks = chunks
        self.corpus_tokens = corpus_tokens

class BM25StoreManager:
    """
    Persistent BM25 Store Manager handling disk serialization and querying
    for 'recursive' and 'semantic' chunking strategies.
    """

    def __init__(self, indexes_dir: Optional[Path] = None):
        self.indexes_dir = indexes_dir or settings.INDEXES_DIR
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

        self._in_memory_indexes: Dict[str, Optional[BM25IndexData]] = {
            "recursive": None,
            "semantic": None
        }

    def _get_filepath(self, strategy: str) -> Path:
        if strategy not in ["recursive", "semantic"]:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be 'recursive' or 'semantic'.")
        return self.indexes_dir / f"bm25_{strategy}.pkl"

    def _load_index(self, strategy: str) -> Optional[BM25IndexData]:
        if self._in_memory_indexes[strategy] is not None:
            return self._in_memory_indexes[strategy]

        filepath = self._get_filepath(strategy)
        if not filepath.is_file():
            return None

        try:
            with open(filepath, "rb") as f:
                index_data: BM25IndexData = pickle.load(f)
            self._in_memory_indexes[strategy] = index_data
            logger.info(f"Loaded persistent BM25 index for '{strategy}' ({len(index_data.chunks)} chunks)")
            return index_data
        except Exception as e:
            logger.error(f"Failed to load BM25 index file {filepath}: {e}")
            return None

    def _save_index(self, index_data: BM25IndexData, strategy: str):
        filepath = self._get_filepath(strategy)
        with open(filepath, "wb") as f:
            pickle.dump(index_data, f)
        self._in_memory_indexes[strategy] = index_data
        logger.info(f"Saved persistent BM25 index for '{strategy}' to {filepath}")

    def add_chunks(self, new_chunks: List[Chunk], strategy: str) -> int:
        if not new_chunks:
            return 0

        existing = self._load_index(strategy)

        if existing is None:
            combined_chunks = list(new_chunks)
            corpus_tokens = [tokenize_text(c.text) for c in combined_chunks]
        else:
            # Deduplicate by chunk_id
            existing_ids = {c.chunk_id for c in existing.chunks}
            unique_new = [c for c in new_chunks if c.chunk_id not in existing_ids]

            if not unique_new:
                logger.info(f"No new chunks to add to BM25 index '{strategy}'")
                return 0

            combined_chunks = existing.chunks + unique_new
            new_tokens = [tokenize_text(c.text) for c in unique_new]
            corpus_tokens = existing.corpus_tokens + new_tokens

        # Re-build BM25 index
        bm25_model = BM25Okapi(corpus_tokens)
        index_data = BM25IndexData(bm25=bm25_model, chunks=combined_chunks, corpus_tokens=corpus_tokens)

        self._save_index(index_data, strategy)
        return len(new_chunks)

    def search(self, query: str, strategy: str, k: int = 15) -> List[SearchResult]:
        index_data = self._load_index(strategy)
        if index_data is None or not index_data.chunks:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = index_data.bm25.get_scores(query_tokens)

        # Pair scores with chunks and sort descending
        chunk_score_pairs = list(zip(index_data.chunks, scores))
        # Filter out 0 scores for efficiency
        positive_pairs = [(chunk, float(score)) for chunk, score in chunk_score_pairs if score > 0.0]
        positive_pairs.sort(key=lambda x: x[1], reverse=True)

        top_k_pairs = positive_pairs[:k]

        results: List[SearchResult] = []
        for rank_idx, (chunk, score) in enumerate(top_k_pairs):
            meta_dict = chunk.metadata.model_dump()
            results.append(SearchResult(
                document_id=chunk.metadata.document_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=meta_dict,
                bm25_score=score,
                retrieval_method=f"bm25_{strategy}",
                rank=rank_idx + 1
            ))

        return results

    def delete_document(self, document_id: str):
        """Remove all chunks associated with document_id and update BM25 store."""
        for strategy in ["recursive", "semantic"]:
            index_data = self._load_index(strategy)
            if index_data is None:
                continue

            filtered_chunks = []
            filtered_tokens = []

            for chunk, tokens in zip(index_data.chunks, index_data.corpus_tokens):
                if chunk.metadata.document_id != document_id:
                    filtered_chunks.append(chunk)
                    filtered_tokens.append(tokens)

            if len(filtered_chunks) == len(index_data.chunks):
                continue

            if not filtered_chunks:
                self.reset(strategy)
            else:
                bm25_model = BM25Okapi(filtered_tokens)
                new_data = BM25IndexData(bm25=bm25_model, chunks=filtered_chunks, corpus_tokens=filtered_tokens)
                self._save_index(new_data, strategy)

            logger.info(f"Deleted chunks for document '{document_id}' from BM25 index '{strategy}'")

    def get_chunk_count(self, strategy: str) -> int:
        index_data = self._load_index(strategy)
        return len(index_data.chunks) if index_data else 0

    def reset(self, strategy: str):
        filepath = self._get_filepath(strategy)
        if filepath.is_file():
            filepath.unlink()
        self._in_memory_indexes[strategy] = None
        logger.info(f"Reset BM25 index for '{strategy}'")
