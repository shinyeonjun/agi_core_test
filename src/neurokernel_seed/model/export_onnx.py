from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import require_torch
from .mlp import WorldModelConfig, build_model


def export_onnx(checkpoint_path: str | Path, out_path: str | Path, verify: bool = True, dynamic_batch: bool = True) -> dict:
    torch = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = WorldModelConfig.from_dict(checkpoint["config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, config.input_dim, dtype=torch.float32)
    export_kwargs = {
        "input_names": ["input_vector"],
        "output_names": ["prediction"],
        "opset_version": 17,
        "dynamo": False,
    }
    if dynamic_batch:
        export_kwargs["dynamic_axes"] = {"input_vector": {0: "batch"}, "prediction": {0: "batch"}}
    torch.onnx.export(model, dummy, out, **export_kwargs)
    result = {
        "checkpoint": str(checkpoint_path),
        "onnx": str(out),
        "input_dim": config.input_dim,
        "target_dim": config.target_dim,
        "dynamic_batch": dynamic_batch,
    }
    if verify:
        result["verification"] = _verify_with_onnxruntime(model, dummy, out)
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({"config": config.as_dict(), "manifest": checkpoint["manifest"], "dynamic_batch": dynamic_batch}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    result["manifest"] = str(manifest_path)
    return result


def _verify_with_onnxruntime(model, dummy, onnx_path: Path) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"available": False, "reason": "onnxruntime not installed"}
    torch = require_torch()
    with torch.no_grad():
        torch_output = model(dummy).detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input_vector": dummy.cpu().numpy()})[0]
    max_abs_diff = float(abs(torch_output - onnx_output).max())
    return {"available": True, "max_abs_diff": max_abs_diff}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="export-world-model-onnx")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="artifacts/world_model.onnx")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--static-batch", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(export_onnx(args.checkpoint, args.out, verify=not args.no_verify, dynamic_batch=not args.static_batch), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
