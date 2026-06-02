from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from agent import __version__
from agent.config.defaults import env_int, now_kst
from agent.dashboard.snapshot import dashboard_snapshot

STATIC_DIR = Path(__file__).parent / "static"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server_version = "AgentCoreDashboard/0.16"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"ok": True, "version": __version__, "created_at": now_kst()})
            return
        if parsed.path == "/api/snapshot":
            query = parse_qs(parsed.query)
            limit = _safe_int(query.get("limit", [None])[0], env_int("AGENT_DASHBOARD_ITEM_LIMIT", 12))
            self._send_json(dashboard_snapshot(limit=limit))
            return
        if parsed.path == "/api/processes":
            query = parse_qs(parsed.query)
            limit = _safe_int(query.get("limit", [None])[0], env_int("AGENT_DASHBOARD_ITEM_LIMIT", 12))
            self._send_json(dashboard_snapshot(limit=limit).get("processes", {}))
            return
        if parsed.path in {"", "/"}:
            self._send_file(STATIC_DIR / "index.html")
            return
        if parsed.path.startswith("/static/"):
            candidate = (STATIC_DIR / parsed.path.removeprefix("/static/")).resolve()
            if STATIC_DIR.resolve() not in candidate.parents and candidate != STATIC_DIR.resolve():
                self._send_error(403, "forbidden")
                return
            self._send_file(candidate)
            return
        self._send_error(404, "not_found")

    def _send_json(self, value: Any, status: int = 200) -> None:
        body = _json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_error(404, "not_found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        if path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        if path.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, reason: str) -> None:
        self._send_json({"ok": False, "reason": reason}, status=status)


def _safe_int(value: object, default: int) -> int:
    try:
        return max(1, min(100, int(str(value))))
    except (TypeError, ValueError):
        return default


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), DashboardRequestHandler)
    print(f"agent-core dashboard listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    run_server(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
