from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
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
            raw = self._read_prompt("\n> ")
            if raw is None:
                return 0
            if not raw:
                continue
            action = menu.dashboard_action(raw)
            if action == "exit":
                print("exit")
                return 0
            if action is None:
                print("failed: unknown command")
                self._wait_for_enter()
                continue
            if not self._run_menu_action(args, action, raw):
                continue

    def print_dashboard(self, args: argparse.Namespace) -> None:
        print()
        print(self._style("NK", "cyan") + "  training automation")
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
            print(f"failed: {exc}", file=sys.stderr)
            self._wait_for_enter()
            return False
        self._print_result(menu_args.action, result)
        self._wait_for_enter()
        return True

    def _print_command_grid(self) -> None:
        for command in menu.DASHBOARD_COMMANDS:
            print(f"{command.key}  {self._style(command.label, 'bold'):<10} {command.hint}")
        print()
        print("quick  learn | cycle | compare | deploy | exit")

    def _print_status_snapshot(self, args: argparse.Namespace) -> None:
        current = self._probe(args, "current")
        best = self._probe(args, "top")
        print(self._kv("world", self._short_current(current)))
        print(self._kv("best", self._short_best(best)))
        slots = current.get("model_slots") if isinstance(current, dict) else {}
        runtime = slots.get("runtime") if isinstance(slots, dict) and isinstance(slots.get("runtime"), dict) else {}
        if runtime:
            print(self._kv("runtime", f"{runtime.get('status', 'unknown')} {runtime.get('model') or ''}".strip()))

    def _probe(self, args: argparse.Namespace, action: str) -> dict[str, Any] | None:
        try:
            return self._run_action(self._make_status_args(args, action))
        except self._model_error_type:
            return None

    def _ask_for_missing_args(self, args: argparse.Namespace) -> None:
        if args.action in {"check", "train"} and not getattr(args, "run_name", None):
            args.run_name = self._optional_prompt("run name")
        elif args.action in {"deploy", "deploy-use", "use"} and not getattr(args, "run_name", None):
            args.run_name = self._required_prompt("run name")
        elif args.action == "bench" and not getattr(args, "run_name", None) and not getattr(args, "model", None):
            raw = self._required_prompt("model/run")
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
            raise self._model_error_type(f"{label} required")
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
            input("\nEnter to continue")
        except EOFError:
            return
