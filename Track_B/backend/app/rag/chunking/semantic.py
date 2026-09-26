import hashlib
import re
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import Chunk, ChunkMetadata, Document

class SemanticChunker:
    """
    Sentence-based Semantic Chunker grouping sentences by embedding distance.
    Identifies semantic breakpoints based on distance percentiles while enforcing
    minimum and maximum character size bounds.
    """

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        breakpoint_percentile: Optional[float] = None,
        buffer_size: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
    ):
        self.model_name = embedding_model_name or settings.EMBEDDING_MODEL
        self.breakpoint_percentile = breakpoint_percentile or settings.SEMANTIC_BREAKPOINT_PERCENTILE
        self.buffer_size = buffer_size or settings.SEMANTIC_BUFFER_SIZE
        self.min_chunk_size = min_chunk_size or settings.SEMANTIC_MIN_CHUNK_SIZE
        self.max_chunk_size = max_chunk_size or settings.SEMANTIC_MAX_CHUNK_SIZE
        
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading sentence transformer for semantic chunker: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex boundary matching."""
        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+')
        raw_sentences = sentence_endings.split(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = np.dot(a, b) / (norm_a * norm_b)
        return float(1.0 - similarity)

    def _combine_sentences_with_buffer(self, sentences: List[str]) -> List[str]:
        """Create buffered sentence representations for smooth embedding comparison."""
        buffered = []
        n = len(sentences)
        for i in range(n):
            start = max(0, i - self.buffer_size)
            end = min(n, i + self.buffer_size + 1)
            buffered.append(" ".join(sentences[start:end]))
        return buffered

    def chunk_document(self, doc: Document) -> List[Chunk]:
        sentences = self._split_into_sentences(doc.text)
        if not sentences:
            return []

        if len(sentences) == 1 or len(doc.text) <= self.min_chunk_size:
            # Single sentence or small doc
            text = doc.text.strip()
            chunk_id = hashlib.sha256(f"{doc.metadata.document_id}_semantic_0_{text[:32]}".encode()).hexdigest()[:24]
            meta = ChunkMetadata(
                document_id=doc.metadata.document_id,
                chunk_id=chunk_id,
                filename=doc.metadata.filename,
                file_type=doc.metadata.file_type,
                file_size=doc.metadata.file_size,
                page_number=doc.metadata.page_number,
                total_pages=doc.metadata.total_pages,
                source=doc.metadata.source,
                title=doc.metadata.title,
                chunk_index=0,
                chunking_strategy="semantic",
                character_count=len(text)
            )
            return [Chunk(chunk_id=chunk_id, text=text, metadata=meta)]

        # Get embeddings for buffered sentences
        buffered_sentences = self._combine_sentences_with_buffer(sentences)
        embeddings = self.model.encode(buffered_sentences, show_progress_bar=False)

        # Compute adjacent distances
        distances = []
        for i in range(len(embeddings) - 1):
            dist = self._cosine_distance(embeddings[i], embeddings[i + 1])
            distances.append(dist)

        # Determine distance threshold from percentile
        if distances:
            threshold = float(np.percentile(distances, self.breakpoint_percentile))
        else:
            threshold = 0.5

        # Build chunk groups based on distance threshold and min/max constraints
        chunk_groups: List[List[str]] = []
        current_group: List[str] = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(len(distances)):
            next_sentence = sentences[i + 1]
            next_len = len(next_sentence)

            is_breakpoint = distances[i] >= threshold
            exceeds_max = (current_len + next_len) > self.max_chunk_size
            below_min = current_len < self.min_chunk_size

            if (is_breakpoint or exceeds_max) and not below_min:
                chunk_groups.append(current_group)
                current_group = [next_sentence]
                current_len = next_len
            else:
                current_group.append(next_sentence)
                current_len += next_len + 1

        if current_group:
            chunk_groups.append(current_group)

        chunks: List[Chunk] = []
        base_doc_id = doc.metadata.document_id

        for idx, group in enumerate(chunk_groups):
            chunk_text = " ".join(group).strip()
            if not chunk_text:
                continue

            prefix = chunk_text[:32].encode("utf-8", errors="ignore")
            raw_id = f"{base_doc_id}_semantic_{idx}_{prefix}".encode("utf-8")
            chunk_id = hashlib.sha256(raw_id).hexdigest()[:24]

            chunk_meta = ChunkMetadata(
                document_id=base_doc_id,
                chunk_id=chunk_id,
                filename=doc.metadata.filename,
                file_type=doc.metadata.file_type,
                file_size=doc.metadata.file_size,
                page_number=doc.metadata.page_number,
                total_pages=doc.metadata.total_pages,
                source=doc.metadata.source,
                title=doc.metadata.title,
                chunk_index=idx,
                chunking_strategy="semantic",
                character_count=len(chunk_text)
            )

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=chunk_meta
            ))

        return chunks
