import hashlib
from typing import List, Optional
from app.core.config import settings
from app.models.schemas import Chunk, ChunkMetadata, Document

class RecursiveChunker:
    """
    Recursive Character Chunker splitting text using tiered separators
    (\\n\\n, \\n, period+space, space, empty string).
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size or settings.RECURSIVE_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RECURSIVE_CHUNK_OVERLAP
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)
        
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            item = split if separator == "" else split + separator
            item_len = len(item)

            if current_length + item_len > self.chunk_size and current_chunk:
                joined = "".join(current_chunk).strip()
                if joined:
                    chunks.append(joined)
                
                # Apply overlap by rewinding
                overlap_size = 0
                overlap_items = []
                for prev_item in reversed(current_chunk):
                    if overlap_size + len(prev_item) <= self.chunk_overlap:
                        overlap_items.insert(0, prev_item)
                        overlap_size += len(prev_item)
                    else:
                        break
                
                current_chunk = overlap_items
                current_length = sum(len(x) for x in current_chunk)

            if len(item) > self.chunk_size and new_separators:
                sub_splits = self._split_text(item, new_separators)
                chunks.extend(sub_splits)
            else:
                current_chunk.append(item)
                current_length += item_len

        if current_chunk:
            final_chunk = "".join(current_chunk).strip()
            if final_chunk:
                chunks.append(final_chunk)

        return chunks

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Convert a Document into a list of Chunk objects with stable chunk IDs."""
        text_chunks = self._split_text(doc.text, self.separators)
        chunks: List[Chunk] = []

        base_doc_id = doc.metadata.document_id

        for idx, text_segment in enumerate(text_chunks):
            # Compute stable chunk ID using doc_id + strategy + index + content prefix
            prefix = text_segment[:32].encode("utf-8", errors="ignore")
            raw_id = f"{base_doc_id}_recursive_{idx}_{prefix}".encode("utf-8")
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
                chunking_strategy="recursive",
                character_count=len(text_segment)
            )

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=text_segment,
                metadata=chunk_meta
            ))

        return chunks
