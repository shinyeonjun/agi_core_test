from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class CoreClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class CoreClient:
    base_url: str
    timeout_seconds: float = 60.0

    def get(self, path: str) -> dict[str, Any] | list[dict[str, Any]]:
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        return self._request("POST", path, payload or {})

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(detail)
            except json.JSONDecodeError:
                message = detail
            else:
                payload = parsed.get("detail") if isinstance(parsed, dict) else parsed
                if isinstance(payload, dict):
                    message = str(payload.get("message") or payload.get("error") or payload)
                else:
                    message = str(payload)
            raise CoreClientError(f"Core API HTTP {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise CoreClientError(f"Core API connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise CoreClientError("Core API request timed out") from exc
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise CoreClientError(f"Core API returned non-JSON response: {body[:300]}") from exc
