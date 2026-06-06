from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a NeuroKernel ONNX world model to RKNN for RK3588 NPU inference.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-platform", default="rk3588")
    parser.add_argument("--quantize", action="store_true", help="Enable RKNN quantization. Keep disabled for first parity checks.")
    args = parser.parse_args()

    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise SystemExit("rknn-toolkit2 is not installed. Install Rockchip RKNN-Toolkit2 on a supported Linux/Python environment first.") from exc

    onnx_path = Path(args.onnx)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(target_platform=args.target_platform)
        _check(ret, "config")
        ret = rknn.load_onnx(model=str(onnx_path))
        _check(ret, "load_onnx")
        ret = rknn.build(do_quantization=bool(args.quantize))
        _check(ret, "build")
        ret = rknn.export_rknn(str(out_path))
        _check(ret, "export_rknn")
    finally:
        rknn.release()

    manifest = onnx_path.with_suffix(".manifest.json")
    manifest_out = out_path.with_suffix(".manifest.json")
    if manifest.exists() and manifest.resolve() != manifest_out.resolve():
        shutil.copy2(manifest, manifest_out)
    print(f"RKNN exported: {out_path}")
    return 0


def _check(code: int, step: str) -> None:
    if code != 0:
        raise SystemExit(f"RKNN {step} failed with code {code}")


if __name__ == "__main__":
    raise SystemExit(main())
