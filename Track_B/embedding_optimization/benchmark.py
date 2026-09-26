import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

def benchmark_embedding_models(onnx_dir: str = "./data/onnx_model"):
    sentences = [
        "Enterprise RAG & AI Assistant System Architecture.",
        "Sentence transformers all-MiniLM-L6-v2 embedding model quantization.",
        "Reciprocal rank fusion hybrid vector lexical retrieval search.",
        "Local vLLM Llama 3.2 1B Instruct AWQ 4GB VRAM GPU memory tuning.",
        "Streamlit user interface with streaming text responses and fallback indicators."
    ] * 20  # 100 sentences batch benchmark

    print("--- Embedding Benchmark: PyTorch FP32 vs ONNX FP32 vs ONNX INT8 ---\n")

    # 1. PyTorch FP32
    print("Benchmarking PyTorch FP32...")
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    start_t = time.perf_counter()
    for _ in range(5):
        _ = st_model.encode(sentences, show_progress_bar=False)
    pytorch_lat = ((time.perf_counter() - start_t) / 5) * 1000.0
    print(f"PyTorch FP32 Avg Latency: {pytorch_lat:.2f} ms")

    # 2. ONNX FP32
    onnx_path = Path(onnx_dir) / "model.onnx"
    if onnx_path.is_file():
        print("Benchmarking ONNX FP32...")
        tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        session_fp32 = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="np")

        start_t = time.perf_counter()
        for _ in range(5):
            ort_inputs = {k: v for k, v in inputs.items() if k in [inp.name for inp in session_fp32.get_inputs()]}
            _ = session_fp32.run(None, ort_inputs)
        onnx_fp32_lat = ((time.perf_counter() - start_t) / 5) * 1000.0
        print(f"ONNX FP32 Avg Latency: {onnx_fp32_lat:.2f} ms")
    else:
        onnx_fp32_lat = 0.0

    # 3. ONNX INT8
    onnx_int8_path = Path(onnx_dir) / "model_int8.onnx"
    if onnx_int8_path.is_file():
        print("Benchmarking ONNX INT8...")
        tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        session_int8 = ort.InferenceSession(str(onnx_int8_path), providers=["CPUExecutionProvider"])
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="np")

        start_t = time.perf_counter()
        for _ in range(5):
            ort_inputs = {k: v for k, v in inputs.items() if k in [inp.name for inp in session_int8.get_inputs()]}
            _ = session_int8.run(None, ort_inputs)
        onnx_int8_lat = ((time.perf_counter() - start_t) / 5) * 1000.0
        print(f"ONNX INT8 Avg Latency: {onnx_int8_lat:.2f} ms")
    else:
        onnx_int8_lat = 0.0

    print("\n--- Summary Benchmark Results ---")
    print(f"PyTorch FP32: {pytorch_lat:.2f} ms")
    if onnx_fp32_lat:
        print(f"ONNX FP32:    {onnx_fp32_lat:.2f} ms (Speedup: {pytorch_lat/onnx_fp32_lat:.2f}x)")
    if onnx_int8_lat:
        print(f"ONNX INT8:    {onnx_int8_lat:.2f} ms (Speedup: {pytorch_lat/onnx_int8_lat:.2f}x)")

if __name__ == "__main__":
    benchmark_embedding_models()
