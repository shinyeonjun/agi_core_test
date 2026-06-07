from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurokernel_seed.eval.gate_ablation import GateAblationConfig, run_gate_ablation
from neurokernel_seed.model.pipeline import TrainingPipelineConfig, run_training_pipeline
from neurokernel_seed.model.release import (
    ModelReleaseError,
    ModelReleaseRemoteConfig,
    TrainDeployModelConfig,
    activate_model_release,
    current_model_release,
    deploy_training_run,
    list_model_releases,
    train_deploy_model,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.action is None:
        return _run_menu(args)
    try:
        result = _run_action(args)
    except ModelReleaseError as exc:
        print(f"실패: {exc}", file=sys.stderr)
        return 1
    _print_result(args.action, result, json_mode=args.json)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nk",
        description="NeuroKernel 모델 학습/배포 리모컨. 그냥 nk만 실행하면 메뉴가 뜬다.",
    )
    sub = parser.add_subparsers(dest="action")
    for name in ("check", "train"):
        p = sub.add_parser(name)
        p.add_argument("run_name", nargs="?", help="비우면 자동 이름 사용")
        _add_train_options(p)
    p = sub.add_parser("deploy")
    p.add_argument("run_name", help="배포할 로컬 training run 이름")
    _add_remote_options(p)
    p.add_argument("--run-dir", help="training run 루트. 기본값은 NEUROKERNEL_TRAIN_RUN_DIR")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("deploy-use")
    p.add_argument("run_name", help="배포 후 현재 모델로 활성화할 로컬 training run 이름")
    _add_remote_options(p)
    p.add_argument("--run-dir", help="training run 루트. 기본값은 NEUROKERNEL_TRAIN_RUN_DIR")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("bench")
    p.add_argument("run_name", nargs="?", help="학습 run 이름. --model을 쓰면 생략 가능")
    p.add_argument("--model", help="직접 검증할 ONNX 모델 경로")
    p.add_argument("--out-dir", help="벤치 결과 저장 위치")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--trace-episodes", type=int, default=10)
    p.add_argument("--max-failures-per-env", type=int, default=10)
    p.add_argument("--no-strict", dest="strict", action="store_false")
    p.set_defaults(strict=True)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("bench-current")
    _add_remote_options(p)
    p.add_argument("--out-dir", help="현재 모델 벤치 저장 위치")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--trace-episodes", type=int, default=10)
    p.add_argument("--max-failures-per-env", type=int, default=10)
    p.add_argument("--no-strict", dest="strict", action="store_false")
    p.set_defaults(strict=True)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("top")
    p.add_argument("--run-dir", help="training run 루트. 기본값은 NEUROKERNEL_TRAIN_RUN_DIR")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("list")
    _add_remote_options(p)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("current")
    _add_remote_options(p)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("use")
    p.add_argument("run_name")
    _add_remote_options(p)
    p.add_argument("--json", action="store_true")
    return parser


def _add_train_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features")
    parser.add_argument("--out-dir")
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-project")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--skip-gate-ablation", action="store_true")
    parser.add_argument("--overwrite-local-run", action="store_true")
    parser.add_argument("--json", action="store_true")


def _add_remote_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-project")
    parser.add_argument("--ssh-connect-timeout", type=int, default=10)


def _run_menu(args: argparse.Namespace) -> int:
    print()
    print("NeuroKernel 모델 리모컨")
    print("1. 학습 전 확인")
    print("2. 학습 + 냉정 벤치 저장")
    print("3. 벤치 상위 모델 비교")
    print("4. 냉정 벤치만 다시 실행")
    print("5. 현재 코어 모델 벤치 저장")
    print("6. 선택 모델 오렌지파이 배포")
    print("7. 선택 모델 배포 후 현재 모델로 활성화")
    print("8. 오렌지파이 모델 목록")
    print("9. 현재 활성 모델 확인")
    print("10. 기존 배포 모델 활성화")
    print("0. 종료")
    choice = input("> 번호 선택: ").strip()
    action_by_choice = {
        "1": "check",
        "2": "train",
        "3": "top",
        "4": "bench",
        "5": "bench-current",
        "6": "deploy",
        "7": "deploy-use",
        "8": "list",
        "9": "current",
        "10": "use",
        "0": "exit",
    }
    action = action_by_choice.get(choice)
    if action is None:
        print("알 수 없는 번호야.")
        return 2
    if action == "exit":
        print("종료.")
        return 0
    run_name = None
    if action in {"check", "train"}:
        raw = input("> 모델 이름(비우면 자동 생성): ").strip()
        run_name = raw or None
    elif action in {"deploy", "deploy-use"}:
        run_name = input("> 배포할 로컬 training run 이름: ").strip()
        if not run_name:
            print("모델 이름이 필요해.")
            return 2
    elif action == "bench":
        raw = input("> 검증할 학습 run 이름 또는 ONNX 경로: ").strip()
        if raw.lower().endswith(".onnx"):
            menu_args = argparse.Namespace(**vars(args))
            menu_args.action = action
            menu_args.run_name = None
            menu_args.model = raw
            menu_args.out_dir = None
            menu_args.episodes = 50
            menu_args.trace_episodes = 10
            menu_args.max_failures_per_env = 10
            menu_args.strict = True
            menu_args.json = False
            try:
                result = _run_action(menu_args)
            except ModelReleaseError as exc:
                print(f"실패: {exc}", file=sys.stderr)
                return 1
            _print_result(action, result, json_mode=False)
            return 0
        run_name = raw or None
    elif action == "use":
        run_name = input("> 활성화할 모델 이름: ").strip()
        if not run_name:
            print("모델 이름이 필요해.")
            return 2
    menu_args = argparse.Namespace(**vars(args))
    menu_args.action = action
    menu_args.run_name = run_name
    menu_args.model = None
    menu_args.out_dir = None
    menu_args.run_dir = None
    menu_args.limit = 5
    menu_args.episodes = 50
    menu_args.trace_episodes = 10
    menu_args.max_failures_per_env = 10
    menu_args.strict = True
    menu_args.json = False
    try:
        result = _run_action(menu_args)
    except ModelReleaseError as exc:
        print(f"실패: {exc}", file=sys.stderr)
        return 1
    _print_result(action, result, json_mode=False)
    return 0


def _run_action(args: argparse.Namespace) -> dict[str, Any]:
    if args.action == "check":
        return train_deploy_model(
            TrainDeployModelConfig(
                features=getattr(args, "features", None),
                out_dir=getattr(args, "out_dir", None),
                run_name=getattr(args, "run_name", None),
                remote_host=getattr(args, "remote_host", None),
                remote_project=getattr(args, "remote_project", None),
                dry_run=True,
                activate=False,
                epochs=getattr(args, "epochs", 50),
                batch_size=getattr(args, "batch_size", 1024),
                device=getattr(args, "device", "cuda"),
                patience=getattr(args, "patience", 10),
                run_gate_ablation=not getattr(args, "skip_gate_ablation", False),
                overwrite_local_run=getattr(args, "overwrite_local_run", False),
            )
        )
    if args.action == "train":
        return _run_local_training_action(args)
    if args.action in {"deploy", "deploy-use"}:
        return _run_deploy_action(args, activate=args.action == "deploy-use")
    if args.action == "bench":
        return _run_benchmark_action(args)
    if args.action == "bench-current":
        return _run_benchmark_current_action(args)
    if args.action == "top":
        return _run_top_models(args)
    remote_config = ModelReleaseRemoteConfig(
        remote_host=getattr(args, "remote_host", None),
        remote_project=getattr(args, "remote_project", None),
        ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
    )
    if args.action == "list":
        return list_model_releases(remote_config)
    if args.action == "current":
        return current_model_release(remote_config)
    if args.action == "use":
        return activate_model_release(args.run_name, remote_config)
    raise ModelReleaseError(f"unknown nk action: {args.action}")


def _run_local_training_action(args: argparse.Namespace) -> dict[str, Any]:
    features = getattr(args, "features", None) or os.getenv("NEUROKERNEL_TRAIN_FEATURES")
    if not features:
        raise ModelReleaseError("train에는 --features 또는 NEUROKERNEL_TRAIN_FEATURES가 필요해")
    out_dir = getattr(args, "out_dir", None) or os.getenv("NEUROKERNEL_TRAIN_RUN_DIR", "artifacts/training_runs")
    return run_training_pipeline(
        TrainingPipelineConfig(
            features=features,
            out_dir=out_dir,
            run_name=getattr(args, "run_name", None),
            epochs=getattr(args, "epochs", 50),
            batch_size=getattr(args, "batch_size", 1024),
            device=getattr(args, "device", "cuda"),
            patience=getattr(args, "patience", 10),
            run_gate_ablation=not getattr(args, "skip_gate_ablation", False),
            overwrite=getattr(args, "overwrite_local_run", False),
            strict=True,
        )
    )


def _run_deploy_action(args: argparse.Namespace, *, activate: bool) -> dict[str, Any]:
    run_root = Path(getattr(args, "run_dir", None) or os.getenv("NEUROKERNEL_TRAIN_RUN_DIR", "artifacts/training_runs"))
    run_name = getattr(args, "run_name", None)
    if not run_name:
        raise ModelReleaseError("deploy에는 run 이름이 필요해")
    run_dir = run_root / run_name
    if not run_dir.exists():
        raise ModelReleaseError(f"training run not found: {run_dir}")
    remote_host = getattr(args, "remote_host", None) or os.getenv("NEUROKERNEL_EDGE_HOST", "orangepi5")
    remote_project = getattr(args, "remote_project", None) or os.getenv("NEUROKERNEL_EDGE_PROJECT", "/home/ubuntu/projects/neurokernel-agi-seed")
    release = deploy_training_run(
        run_dir,
        release_name=run_name,
        remote_host=remote_host,
        remote_project=remote_project,
        activate=activate,
        ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
    )
    return {
        "status": "deployed_and_activated" if activate else "deployed",
        "run_name": run_name,
        "run_dir": str(run_dir),
        "release": release,
    }


def _run_benchmark_action(args: argparse.Namespace) -> dict[str, Any]:
    model_path = _resolve_benchmark_model(args)
    out_dir = getattr(args, "out_dir", None)
    if out_dir:
        run_dir = Path(out_dir)
        benchmark_dir = run_dir / "gate_ablation"
    else:
        run_dir = model_path.parent
        benchmark_dir = run_dir / f"gate_ablation_nk_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    result = run_gate_ablation(
        model_path,
        out_dir=benchmark_dir,
        config=GateAblationConfig(
            episodes=getattr(args, "episodes", 50),
            trace_episodes=getattr(args, "trace_episodes", 10),
            max_failures_per_env=getattr(args, "max_failures_per_env", 10),
            strict=getattr(args, "strict", True),
        ),
    )
    (run_dir / "gate_ablation_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _run_benchmark_current_action(args: argparse.Namespace) -> dict[str, Any]:
    remote_config = ModelReleaseRemoteConfig(
        remote_host=getattr(args, "remote_host", None),
        remote_project=getattr(args, "remote_project", None),
        ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
    )
    current = current_model_release(remote_config)
    current_payload = current.get("current") or {}
    release_name = current_payload.get("release_name") or current_payload.get("run_name") or "current_core"
    safe_release_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(release_name))
    benchmark_dir = Path(getattr(args, "out_dir", None) or _default_current_bench_dir(safe_release_name))
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    remote_host = current["remote_host"]
    remote_project = current["remote_project"].rstrip("/")
    _copy_remote_file(remote_host, f"{remote_project}/artifacts/current_world_model.onnx", benchmark_dir / "world_model.onnx")
    _copy_remote_file(
        remote_host,
        f"{remote_project}/artifacts/current_world_model.manifest.json",
        benchmark_dir / "world_model.manifest.json",
    )
    result = run_gate_ablation(
        benchmark_dir / "world_model.onnx",
        out_dir=benchmark_dir / "gate_ablation",
        config=GateAblationConfig(
            episodes=getattr(args, "episodes", 50),
            trace_episodes=getattr(args, "trace_episodes", 10),
            max_failures_per_env=getattr(args, "max_failures_per_env", 10),
            strict=getattr(args, "strict", True),
        ),
    )
    result_path = benchmark_dir / "gate_ablation_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "status": "completed",
        "source": "orangepi_current_model",
        "release_name": release_name,
        "remote_host": remote_host,
        "remote_project": remote_project,
        "run_dir": str(benchmark_dir),
        "gate_ablation_result": str(result_path),
    }
    (benchmark_dir / "current_core_benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest | {"benchmark": result}


def _default_current_bench_dir(release_name: str) -> Path:
    run_root = Path(os.getenv("NEUROKERNEL_TRAIN_RUN_DIR", "artifacts/training_runs"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return run_root / f"current_core_{release_name}_{stamp}"


def _copy_remote_file(remote_host: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["scp", f"{remote_host}:{remote_path}", str(local_path)], check=False, text=True, capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ModelReleaseError(f"scp failed for {remote_path}: {detail}")


def _resolve_benchmark_model(args: argparse.Namespace) -> Path:
    explicit_model = getattr(args, "model", None)
    if explicit_model:
        model_path = Path(explicit_model)
    else:
        run_name = getattr(args, "run_name", None)
        if not run_name:
            raise ModelReleaseError("bench에는 run 이름이나 --model 경로가 필요해")
        run_root = Path(os.getenv("NEUROKERNEL_TRAIN_RUN_DIR", "artifacts/training_runs"))
        model_path = run_root / run_name / "world_model.onnx"
    if not model_path.exists():
        raise ModelReleaseError(f"benchmark model not found: {model_path}")
    return model_path


def _run_top_models(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(getattr(args, "run_dir", None) or os.getenv("NEUROKERNEL_TRAIN_RUN_DIR", "artifacts/training_runs"))
    limit = max(1, int(getattr(args, "limit", 5)))
    candidates = [_summarize_training_run(path) for path in sorted(run_root.glob("*/gate_ablation_result.json"))]
    candidates = [item for item in candidates if item is not None]
    ranked = sorted(candidates, key=_top_sort_key, reverse=True)
    return {
        "status": "ok",
        "run_root": str(run_root),
        "limit": limit,
        "count": len(candidates),
        "ranking_basis": [
            "macro_success_rate.hybrid_veto",
            "model_needed_signal_group_count",
            "hybrid_veto_gain_over_prior",
        ],
        "top": ranked[:limit],
    }


def _summarize_training_run(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    aggregate = payload.get("aggregate") or {}
    macro = aggregate.get("macro_success_rate") or {}
    interpretation = payload.get("interpretation") or {}
    return {
        "run_name": path.parent.name,
        "report": str(path),
        "verdict": interpretation.get("verdict"),
        "hybrid_veto_success": _as_float(macro.get("hybrid_veto")),
        "hybrid_success": _as_float(macro.get("hybrid")),
        "prior_success": _as_float(macro.get("prior_only")),
        "model_only_success": _as_float(macro.get("model_only")),
        "model_needed_signal_group_count": int(aggregate.get("model_needed_signal_group_count") or 0),
        "hybrid_veto_gain_over_prior": _as_float(aggregate.get("hybrid_veto_gain_over_prior")),
        "hybrid_gain_over_prior": _as_float(aggregate.get("hybrid_gain_over_prior")),
        "groups_with_model_needed_signal": aggregate.get("groups_with_model_needed_signal") or [],
    }


def _top_sort_key(item: dict[str, Any]) -> tuple[float, int, float]:
    return (
        item["hybrid_veto_success"],
        item["model_needed_signal_group_count"],
        item["hybrid_veto_gain_over_prior"],
    )


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _print_result(action: str, result: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if action == "check":
        preflight = result["preflight"]
        print("확인 완료. 학습 들어가도 되는 상태야.")
        print(f"- 모델 이름: {result['run_name']}")
        print(f"- 로컬 저장 위치: {preflight['local_run_dir']}")
        print(f"- 오렌지파이 위치: {preflight['remote_release_dir']}")
        return
    if action == "train":
        print("학습과 냉정 벤치가 끝났어. 아직 오렌지파이에 배포하진 않았어.")
        print(f"- 모델 이름: {result['run_name']}")
        print(f"- 로컬 결과 위치: {result['run_dir']}")
        if result.get("artifacts", {}).get("gate_ablation"):
            print(f"- 벤치 결과: {result['artifacts']['gate_ablation']}")
        return
    if action == "deploy":
        print("선택한 모델을 오렌지파이에 배포했어. 아직 현재 모델로 활성화하진 않았어.")
        print(f"- 모델 이름: {result['run_name']}")
        print(f"- 오렌지파이 위치: {result['release']['remote_release_dir']}")
        return
    if action == "deploy-use":
        print("선택한 모델을 배포하고 현재 모델로 활성화했어.")
        print(f"- 현재 모델: {result['run_name']}")
        return
    if action == "bench":
        interpretation = result.get("interpretation", {})
        print("냉정 벤치가 끝났어.")
        if interpretation.get("verdict"):
            print(f"- 판정: {interpretation['verdict']}")
        if interpretation.get("summary"):
            print(f"- 요약: {interpretation['summary']}")
        if result.get("out"):
            print(f"- 결과 파일: {result['out']}")
        return
    if action == "bench-current":
        benchmark = result.get("benchmark", {})
        interpretation = benchmark.get("interpretation", {})
        print("현재 코어 모델 벤치 결과를 저장했어.")
        print(f"- 모델 이름: {result['release_name']}")
        print(f"- 저장 위치: {result['run_dir']}")
        if interpretation.get("verdict"):
            print(f"- 판정: {interpretation['verdict']}")
        return
    if action == "top":
        top = result.get("top") or []
        if not top:
            print("비교할 벤치 결과가 아직 없어.")
            print(f"- 확인한 위치: {result['run_root']}")
            return
        print("벤치 상위 모델")
        print("기준: hybrid_veto 성공률 > model-needed 신호 수 > prior 대비 이득")
        for index, item in enumerate(top, start=1):
            print(
                f"{index}. {item['run_name']} | "
                f"hv={item['hybrid_veto_success']:.4f}, "
                f"prior={item['prior_success']:.4f}, "
                f"gain={item['hybrid_veto_gain_over_prior']:.4f}, "
                f"signal={item['model_needed_signal_group_count']}, "
                f"verdict={item.get('verdict') or 'unknown'}"
            )
        return
    if action == "list":
        releases = result["releases"]
        if not releases:
            print("오렌지파이에 저장된 모델 릴리즈가 아직 없어.")
            return
        print("오렌지파이 모델 목록")
        for release_name in releases:
            print(f"- {release_name}")
        return
    if action == "current":
        current = result.get("current") or {}
        release_name = current.get("release_name") or current.get("run_name") or "(이름 없음)"
        print(f"현재 활성 모델: {release_name}")
        if current.get("deployed_at"):
            print(f"- 배포 시간: {current['deployed_at']}")
        if current.get("remote_release_dir"):
            print(f"- 위치: {current['remote_release_dir']}")
        return
    if action == "use":
        print(f"현재 모델을 바꿨어: {result['release_name']}")


if __name__ == "__main__":
    raise SystemExit(main())
