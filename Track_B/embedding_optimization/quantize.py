from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(input_onnx_path: str = "./data/onnx_model/model.onnx", output_onnx_path: str = "./data/onnx_model/model_int8.onnx"):
    in_path = Path(input_onnx_path)
    out_path = Path(output_onnx_path)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input ONNX model not found at {input_onnx_path}. Run export_onnx.py first.")

    print(f"Applying INT8 dynamic quantization to: {in_path}...")
    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QUInt8
    )

    print(f"INT8 quantized model saved successfully to: {out_path}")
    print(f"Original FP32 ONNX Size: {in_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Quantized INT8 ONNX Size: {out_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    quantize_onnx_model()
