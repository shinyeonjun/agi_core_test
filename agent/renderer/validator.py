from __future__ import annotations

FORBIDDEN_PHRASES = (
    "\uc790\ub3d9 sudo \uc2e4\ud589",
    "sudo \uc790\ub3d9 \uc2e4\ud589",
    "rm -rf",
    "/etc \uc218\uc815 \uc790\ub3d9 \uc2e4\ud589",
    "/etc/",
    "SSH \ud0a4 \uc811\uadfc",
    "ssh key",
    "\uc2b9\uc778 \uc5c6\uc774",
    "\uc758\uc2dd\uc774 \uc0dd\uae40",
    "\ub8e8\ud2b8 \uad8c\ud55c\uc73c\ub85c \ucc98\ub9ac",
)


def validate_output(text: str, must_include: list[str] | None = None, must_not_include: list[str] | None = None) -> dict[str, object]:
    missing = [item for item in (must_include or []) if item not in text]
    forbidden = [item for item in FORBIDDEN_PHRASES if item in text]
    forbidden.extend([item for item in (must_not_include or []) if item in text])
    return {
        "ok": not missing and not forbidden,
        "missing": missing,
        "forbidden": sorted(set(forbidden)),
    }
