#!/usr/bin/env bash
set -e

# ==============================================================================
# Local vLLM Model Server Script tuned for 4GB VRAM (RTX 3050)
# ==============================================================================
# Hardware constraint tuning:
# 1. gpu_memory_utilization: 0.45 (Allocates ~1.8GB VRAM for weights + KV cache,
#    leaving ~2.2GB for CUDA runtime & PyTorch context).
# 2. max_model_len: 2048 (Reduces KV cache memory footprint drastically).
# 3. max_num_seqs: 8 (Limits concurrent sequence processing to preserve VRAM).
# 4. enforce_eager: Disables CUDA graph capture, saving ~500MB VRAM.
# ==============================================================================

MODEL_NAME=${1:-"Qwen/Qwen2.5-1.5B-Instruct-AWQ"}
PORT=${PORT:-8001}
HOST=${HOST:-"0.0.0.0"}
GPU_MEM_UTIL=${GPU_MEMORY_UTILIZATION:-0.45}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}

echo "Starting vLLM OpenAI-Compatible API Server..."
echo "Model: ${MODEL_NAME}"
echo "Host: ${HOST}:${PORT}"
echo "GPU Memory Utilization: ${GPU_MEM_UTIL}"
echo "Max Model Length: ${MAX_MODEL_LEN}"

python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --quantization awq \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs 8 \
    --enforce-eager \
    --trust-remote-code
