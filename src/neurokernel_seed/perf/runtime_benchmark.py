from __future__ import annotations

import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from neurokernel_seed.eval.hard_heldout import build_hard_eval_groups


def benchmark_runtime(
    model_path: str | Path,
    *,
    backend: str = "onnx_cpu",
    iterations: int = 2_000,
    samples: int = 256,
    batches: list[int] | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    model = Path(model_path)
    if batches is None:
        batches = [1, 8, 32, 128]
    payload: dict[str, Any] = {
        "benchmark_version": "neurokernel-runtime-speed-v1",
        "backend": backend,
        "model": str(model),
        "runtime_env": inspect_runtime_env(),
        "iterations": iterations,
        "samples": samples,
        "batches": batches,
    }
    if backend == "onnx_cpu":
        payload["raw_model"] = _benchmark_raw_onnx(model, iterations=iterations, batches=batches)
        payload["predictor_loop"] = _benchmark_predictor(model, backend=backend, iterations=iterations, sample_count=samples)
    elif backend == "rknn_npu":
        payload["rknn_status"] = _rknn_status(model)
        if not payload["rknn_status"]["available"]:
            payload["available"] = False
            payload["reason"] = payload["rknn_status"]["reason"]
        else:
            payload["available"] = True
            payload["predictor_loop"] = _benchmark_predictor(model, backend=backend, iterations=iterations, sample_count=samples)
    else:
        raise ValueError(f"unknown backend: {backend}")
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        payload["out"] = str(out_path)
    return payload


def inspect_runtime_env() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "onnxruntime": _module_available("onnxruntime"),
        "rknnlite": _module_available("rknnlite"),
        "rknn_toolkit_lite2": _module_available("rknn_toolkit_lite2"),
        "rknn": _module_available("rknn"),
    }


def _benchmark_raw_onnx(model: Path, *, iterations: int, batches: list[int]) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    meta = _load_manifest(model)
    input_dim = int(meta["config"]["input_dim"])
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    results: dict[str, Any] = {}
    for batch in batches:
        input_array = rng.normal(0.0, 1.0, size=(batch, input_dim)).astype("float32")
        for _ in range(20):
            session.run(None, {"input_vector": input_array})
        timings = []
        loops = max(10, iterations // max(1, batch))
        for _ in range(loops):
            start = time.perf_counter()
            session.run(None, {"input_vector": input_array})
            timings.append(time.perf_counter() - start)
        per_batch = _summarize_seconds(timings)
        results[str(batch)] = {
            **per_batch,
            "items_per_second_median": batch / per_batch["median_seconds"] if per_batch["median_seconds"] > 0 else None,
            "seconds_per_item_median": per_batch["median_seconds"] / batch,
        }
    return {"provider": "CPUExecutionProvider", "by_batch": results}


def _benchmark_predictor(model: Path, *, backend: str, iterations: int, sample_count: int) -> dict[str, Any]:
    predictor = _make_predictor(model, backend)
    pairs = _collect_state_action_pairs(sample_count)
    for index in range(min(50, len(pairs))):
        state, action = pairs[index]
        predictor.predict(state, action)
    timings = []
    for index in range(iterations):
        state, action = pairs[index % len(pairs)]
        start = time.perf_counter()
        predictor.predict(state, action)
        timings.append(time.perf_counter() - start)
    close = getattr(predictor, "close", None)
    if callable(close):
        close()
    summary = _summarize_seconds(timings)
    return {
        **summary,
        "predictions_per_second_median": 1.0 / summary["median_seconds"] if summary["median_seconds"] > 0 else None,
        "sample_count": len(pairs),
    }


def _collect_state_action_pairs(limit: int):
    pairs = []
    for cases in build_hard_eval_groups().values():
        for case in cases:
            env = case.env
            state = env.reset(0)
            for action in env.candidate_actions(state):
                pairs.append((state, action))
                if len(pairs) >= limit:
                    return pairs
    if not pairs:
        raise RuntimeError("no benchmark samples were generated")
    return pairs


def _make_predictor(model: Path, backend: str):
    if backend == "onnx_cpu":
        from neurokernel_seed.predictors.learned_onnx import OnnxWorldModelPredictor

        return OnnxWorldModelPredictor(model)
    if backend == "rknn_npu":
        from neurokernel_seed.predictors.learned_rknn import RknnWorldModelPredictor

        return RknnWorldModelPredictor(model)
    raise ValueError(backend)


def _rknn_status(model: Path) -> dict[str, Any]:
    if model.suffix.lower() != ".rknn":
        return {"available": False, "reason": "backend rknn_npu expects a .rknn model file"}
    if importlib.util.find_spec("rknnlite") is None:
        return {"available": False, "reason": "rknnlite module is not installed"}
    if not model.exists():
        return {"available": False, "reason": f"model not found: {model}"}
    if not model.with_suffix(".manifest.json").exists():
        return {"available": False, "reason": f"manifest not found: {model.with_suffix('.manifest.json')}"}
    return {"available": True}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load_manifest(model: Path) -> dict[str, Any]:
    manifest_path = model.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _summarize_seconds(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "min_seconds": min(values),
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.fmean(values),
        "p95_seconds": ordered[int((len(ordered) - 1) * 0.95)],
        "max_seconds": max(values),
    }
