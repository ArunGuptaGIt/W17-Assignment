from typing import Any, Dict, List, Optional
import numpy as np
from pathlib import Path
import onnxruntime as ort
from transformers import AutoTokenizer
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import Chunk, SearchResult

class LocalEmbeddingFunction:
    """ChromaDB compatible embedding function supporting PyTorch FP32, ONNX FP32, and ONNX INT8."""

    def __init__(self, model_name: str, backend: str = "pytorch"):
        self.model_name = model_name
        self.backend = backend
        self._model: Optional[SentenceTransformer] = None
        self._onnx_session: Optional[ort.InferenceSession] = None
        self._tokenizer = None

    def _load_onnx(self, model_filename: str):
        onnx_dir = settings.DATA_DIR / "onnx_model"
        model_path = onnx_dir / model_filename
        if not model_path.is_file():
            logger.warning(f"ONNX model '{model_filename}' not found at {model_path}. Falling back to PyTorch FP32.")
            self.backend = "pytorch"
            self._model = SentenceTransformer(self.model_name)
            return

        logger.info(f"Loading ONNX embedding model ({model_filename}) from: {model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
        self._onnx_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading SentenceTransformer embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __call__(self, input: List[str]) -> List[List[float]]:
        if self.backend == "onnx_fp32":
            if self._onnx_session is None:
                self._load_onnx("model.onnx")
        elif self.backend == "onnx_int8":
            if self._onnx_session is None:
                self._load_onnx("model_int8.onnx")

        if self.backend == "pytorch" or self._onnx_session is None:
            embeddings = self.model.encode(input, show_progress_bar=False, normalize_embeddings=True)
            return embeddings.tolist()

        # Run ONNX inference
        inputs = self._tokenizer(input, padding=True, truncation=True, return_tensors="np")
        ort_inputs = {k: v for k, v in inputs.items() if k in [inp.name for inp in self._onnx_session.get_inputs()]}
        outputs = self._onnx_session.run(None, ort_inputs)

        last_hidden_state = outputs[0]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
            sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_embeddings / sum_mask
        else:
            mean_pooled = np.mean(last_hidden_state, axis=1)

        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = mean_pooled / norms
        return normalized.tolist()

class VectorStoreManager:
    """
    ChromaDB Manager creating and querying distinct collections:
    'chunks_recursive' and 'chunks_semantic'.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = str(persist_dir or settings.CHROMA_PERSIST_DIR)
        logger.info(f"Initializing ChromaDB client at: {self.persist_dir}")
        
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.embedding_fn = LocalEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL,
            backend=settings.EMBEDDING_BACKEND
        )

        self.collections = {
            "recursive": self.client.get_or_create_collection(
                name="chunks_recursive",
                metadata={"hnsw:space": "cosine"}
            ),
            "semantic": self.client.get_or_create_collection(
                name="chunks_semantic",
                metadata={"hnsw:space": "cosine"}
            )
        }

    def _get_collection(self, strategy: str):
        if strategy not in self.collections:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be 'recursive' or 'semantic'.")
        return self.collections[strategy]

    def add_chunks(self, chunks: List[Chunk], strategy: str) -> int:
        if not chunks:
            return 0

        collection = self._get_collection(strategy)

        ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata.model_dump() for chunk in chunks]

        # Clean metadata dict for ChromaDB (convert list/dict/None values to primitive types)
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)

        embeddings = self.embedding_fn(texts)

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=clean_metadatas
        )

        logger.info(f"Indexed {len(chunks)} chunks into Chroma collection 'chunks_{strategy}'")
        return len(chunks)

    def similarity_search(self, query: str, strategy: str, k: int = 15) -> List[SearchResult]:
        collection = self._get_collection(strategy)
        if collection.count() == 0:
            return []

        query_embedding = self.embedding_fn([query])

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        search_results: List[SearchResult] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for rank_idx, (chunk_id, doc_text, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
            # Convert cosine distance (0 to 2) to cosine similarity score (0 to 1)
            score = float(1.0 - (dist / 2.0))
            doc_id = meta.get("document_id", "")

            search_results.append(SearchResult(
                document_id=doc_id,
                chunk_id=chunk_id,
                text=doc_text,
                metadata=meta,
                dense_score=score,
                retrieval_method=f"dense_{strategy}",
                rank=rank_idx + 1
            ))

        return search_results

    def delete_document(self, document_id: str):
        """Remove chunks corresponding to document_id from both collections."""
        for name, collection in self.collections.items():
            existing = collection.get(where={"document_id": document_id})
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])
                logger.info(f"Deleted {len(existing['ids'])} chunks for document '{document_id}' from 'chunks_{name}'")

    def get_chunk_count(self, strategy: str) -> int:
        return self._get_collection(strategy).count()

    def reset(self):
        for strategy in ["recursive", "semantic"]:
            name = f"chunks_{strategy}"
            self.client.delete_collection(name)
            self.collections[strategy] = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
        logger.info("Reset all ChromaDB collections.")
