from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str, hint: str | None = None) -> str:
    suffix = uuid4().hex[:12]
    if hint:
        safe_hint = "".join(char if char.islower() or char.isdigit() or char == "_" else "_" for char in hint.lower()).strip("_")
        safe_hint = safe_hint[:36] or "item"
        return f"{prefix}_{safe_hint}_{suffix}"
    return f"{prefix}_{suffix}"
