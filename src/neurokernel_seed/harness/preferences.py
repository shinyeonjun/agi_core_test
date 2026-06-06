from __future__ import annotations

from typing import Any


def preference_map(preferences: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pref in preferences:
        key = str(pref.get("key") or "")
        if not key:
            continue
        result[key] = pref.get("value_json")
    return result
