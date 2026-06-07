from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neurokernel_seed.nk_console import menu


RunAction = Callable[[argparse.Namespace], dict[str, Any]]
PrintResult = Callable[[str, dict[str, Any]], None]
MakeStatusArgs = Callable[[argparse.Namespace, str], argparse.Namespace]
ShortResult = Callable[[dict[str, Any] | None], str]
TextStyle = Callable[[str, str], str]
KeyValue = Callable[[str, Any], str]


class DashboardController:
    """Small interactive wrapper for the nk automation commands."""

    def __init__(
        self,
        *,
        run_action: RunAction,
        print_result: PrintResult,
        make_status_args: MakeStatusArgs,
        model_error_type: type[Exception],
        short_current: ShortResult,
        short_best: ShortResult,
        style: TextStyle,
        kv: KeyValue,
    ) -> None:
        self._run_action = run_action
        self._print_result = print_result
        self._make_status_args = make_status_args
        self._model_error_type = model_error_type
        self._short_current = short_current
        self._short_best = short_best
        self._style = style
        self._kv = kv

    def run(self, args: argparse.Namespace) -> int:
        while True:
            self.print_dashboard(args)
            raw = self._read_prompt("\n선택> ")
            if raw is None:
                return 0
            if not raw:
                continue
            action = menu.dashboard_action(raw)
            if action == "exit":
                print("종료")
                return 0
            if action is None:
                print("알 수 없는 명령입니다. 번호나 빠른 명령을 입력해줘.")
                self._wait_for_enter()
                continue
            if not self._run_menu_action(args, action, raw):
                continue

    def print_dashboard(self, args: argparse.Namespace) -> None:
        print()
        print(self._style("NK", "cyan") + "  NeuroKernel 자동 학습 콘솔")
        print("OrangePi 데이터를 모으고, 노트북 CUDA로 학습한 뒤, 품질 통과 모델만 배포합니다.")
        self._print_command_grid()
        print()
        self._print_status_snapshot(args)

    def _run_menu_action(self, base_args: argparse.Namespace, action: str, raw: str) -> bool:
        menu_args = menu.build_menu_args(base_args, action)
        menu.apply_inline_value(menu_args, raw)
        try:
            self._ask_for_missing_args(menu_args)
            result = self._run_action(menu_args)
        except self._model_error_type as exc:
            print(f"실패: {exc}", file=sys.stderr)
            self._wait_for_enter()
            return False
        self._print_result(menu_args.action, result)
        self._wait_for_enter()
        return True

    def _print_command_grid(self) -> None:
        print()
        print("번호  명령       작업")
        for command in menu.DASHBOARD_COMMANDS:
            print(f"{command.key:<4} {command.label:<10} {command.hint}")
        print()
        print("빠른 입력  오토파일럿 | 시드 | 학습만 | 런타임배포 | 상태 | 종료")
        print("직접 실행  nk 오토파일럿 --device cuda  또는  nk runtime-train --device cuda")

    def _print_status_snapshot(self, args: argparse.Namespace) -> None:
        current = self._probe(args, "current")
        best = self._probe(args, "top")
        local = _local_runtime_snapshot()
        print("현재 상태")
        print(self._kv("world", self._short_current(current)))
        print(self._kv("best", self._short_best(best)))
        slots = current.get("model_slots") if isinstance(current, dict) else {}
        runtime = slots.get("runtime") if isinstance(slots, dict) and isinstance(slots.get("runtime"), dict) else {}
        if runtime:
            print(self._kv("runtime", f"{runtime.get('status', 'unknown')} {runtime.get('model') or ''}".strip()))
        print(self._kv("local-data", local["data"]))
        print(self._kv("local-model", local["model"]))
        print(self._kv("추천", _recommend_next_action(local, runtime)))

    def _probe(self, args: argparse.Namespace, action: str) -> dict[str, Any] | None:
        try:
            return self._run_action(self._make_status_args(args, action))
        except self._model_error_type:
            return None

    def _ask_for_missing_args(self, args: argparse.Namespace) -> None:
        if args.action in {"check", "train"} and not getattr(args, "run_name", None):
            args.run_name = self._optional_prompt("실행 이름")
        elif args.action in {"deploy", "deploy-use", "use"} and not getattr(args, "run_name", None):
            args.run_name = self._required_prompt("실행 이름")
        elif args.action == "bench" and not getattr(args, "run_name", None) and not getattr(args, "model", None):
            raw = self._required_prompt("모델 또는 실행 이름")
            if raw.lower().endswith(".onnx"):
                args.model = raw
            else:
                args.run_name = raw

    def _optional_prompt(self, label: str) -> str | None:
        value = self._read_prompt(f"{label}> ")
        return value or None

    def _required_prompt(self, label: str) -> str:
        value = self._read_prompt(f"{label}> ")
        if not value:
            raise self._model_error_type(f"{label} 필요")
        return value

    @staticmethod
    def _read_prompt(prompt: str) -> str | None:
        try:
            return input(prompt).strip()
        except EOFError:
            return None

    @staticmethod
    def _wait_for_enter() -> None:
        if not sys.stdin.isatty():
            return
        try:
            input("\n계속하려면 Enter")
        except EOFError:
            return


def _local_runtime_snapshot() -> dict[str, str]:
    features = Path(os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
    model = Path(os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt"))
    feature_manifest = _read_json(features.with_suffix(features.suffix + ".manifest.json"))
    model_manifest = _read_json(model.with_suffix(".manifest.json"))
    data_text = "없음"
    if feature_manifest:
        actions = len(feature_manifest.get("action_vocab") or [])
        data_text = f"{feature_manifest.get('rows')} rows / action {actions}"
    model_text = "없음"
    if model.exists() and model_manifest:
        model_text = f"candidate rows={model_manifest.get('rows')} input={model_manifest.get('input_dim')}"
    elif model.exists():
        model_text = "candidate manifest 없음"
    return {"data": data_text, "model": model_text}


def _recommend_next_action(local: dict[str, str], runtime: dict[str, Any]) -> str:
    if local["data"] == "없음":
        return "오토파일럿 또는 시드로 데이터 준비"
    if local["model"] == "없음":
        return "학습만 또는 오토파일럿"
    remote_rows = _remote_runtime_rows(runtime)
    local_rows = _rows_from_text(local["model"])
    if local_rows and remote_rows is not None and local_rows > remote_rows:
        return "런타임배포: 로컬 후보가 원격보다 최신"
    return "오토파일럿: 데이터 갱신 후 품질 통과 시 자동 배포"


def _remote_runtime_rows(runtime: dict[str, Any]) -> int | None:
    manifest = runtime.get("manifest_payload") if isinstance(runtime, dict) else None
    if not isinstance(manifest, dict):
        return None
    try:
        return int(manifest.get("rows"))
    except (TypeError, ValueError):
        return None


def _rows_from_text(value: str) -> int | None:
    if "rows=" not in value:
        return None
    raw = value.split("rows=", 1)[1].split()[0]
    try:
        return int(raw)
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
