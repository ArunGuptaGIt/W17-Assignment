import hashlib
from pathlib import Path
from typing import List, Union
import pypdf

from app.core.logging import logger
from app.models.schemas import Document, DocumentMetadata

def compute_document_id(content_bytes: bytes) -> str:
    """Compute deterministic SHA-256 hash for document content."""
    return hashlib.sha256(content_bytes).hexdigest()

def load_document(file_path: Union[str, Path]) -> List[Document]:
    """
    Load document from file path (.txt, .md, .pdf).
    Returns list of Document instances (one per page for PDFs, one per file for text/md).
    Preserves page numbers and metadata.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    file_bytes = path.read_bytes()
    doc_id = compute_document_id(file_bytes)
    file_size = len(file_bytes)
    filename = path.name

    documents: List[Document] = []

    if ext == ".pdf":
        logger.info(f"Loading PDF document: {filename} (size: {file_size} bytes)")
        reader = pypdf.PdfReader(path)
        total_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            
            page_number = page_idx + 1
            meta = DocumentMetadata(
                document_id=doc_id,
                filename=filename,
                file_type="pdf",
                file_size=file_size,
                page_number=page_number,
                total_pages=total_pages,
                source=str(path.resolve()),
                title=filename
            )
            documents.append(Document(
                document_id=f"{doc_id}_p{page_number}",
                text=text.strip(),
                metadata=meta
            ))
    elif ext in [".txt", ".md"]:
        logger.info(f"Loading text document: {filename} (size: {file_size} bytes)")
        text = file_bytes.decode("utf-8", errors="replace")
        file_type = "markdown" if ext == ".md" else "text"
        meta = DocumentMetadata(
            document_id=doc_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            page_number=1,
            total_pages=1,
            source=str(path.resolve()),
            title=filename
        )
        documents.append(Document(
            document_id=doc_id,
            text=text.strip(),
            metadata=meta
        ))
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .pdf, .txt, .md")

    logger.info(f"Loaded {len(documents)} page/document block(s) for {filename}")
    return documents
