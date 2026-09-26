import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

def export_to_onnx(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", output_dir: str = "./data/onnx_model"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_path = out_path / "model.onnx"

    print(f"Loading PyTorch model: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    tokenizer.save_pretrained(out_path)

    dummy_text = "Enterprise RAG ONNX embedding model export benchmark string."
    inputs = tokenizer(dummy_text, return_tensors="pt")

    input_names = ["input_ids", "attention_mask"]
    output_names = ["last_hidden_state"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
    }

    if "token_type_ids" in inputs:
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}

    print(f"Exporting PyTorch model to ONNX format at: {onnx_path}...")
    torch.onnx.export(
        model,
        tuple(inputs.values()),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True
    )

    print(f"ONNX export successfully created at: {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()
