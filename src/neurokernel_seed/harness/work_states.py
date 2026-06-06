from __future__ import annotations


PROPOSAL_TRANSITIONS: dict[str, set[str]] = {
    "proposed": {"approved_for_dev", "rejected", "deferred", "duplicate", "unsafe"},
    "deferred": {"proposed", "approved_for_dev", "rejected"},
    "approved_for_dev": {"active", "deferred", "rejected"},
    "active": {"archived"},
    "duplicate": set(),
    "unsafe": set(),
    "rejected": set(),
    "archived": set(),
}

WORK_ITEM_TRANSITIONS: dict[str, set[str]] = {
    "proposed": {"accepted", "planned", "running", "waiting_approval", "rejected", "deferred", "cancelled", "completed", "failed"},
    "accepted": {"planned", "running", "waiting_approval", "blocked", "cancelled", "completed", "failed"},
    "planned": {"running", "blocked", "waiting_approval", "cancelled", "completed", "failed"},
    "running": {"blocked", "waiting_approval", "reviewing", "completed", "failed", "cancelled"},
    "blocked": {"planned", "running", "waiting_approval", "cancelled", "failed"},
    "waiting_approval": {"planned", "running", "reviewing", "completed", "rejected", "cancelled"},
    "reviewing": {"running", "completed", "failed", "cancelled"},
    "deferred": {"proposed", "planned", "cancelled"},
    "rejected": set(),
    "completed": {"archived"},
    "failed": {"planned", "archived"},
    "cancelled": {"archived"},
    "archived": set(),
}


def allowed_proposal_next_statuses(current: str) -> set[str]:
    return PROPOSAL_TRANSITIONS.get(current, set())


def allowed_work_next_statuses(current: str) -> set[str]:
    return WORK_ITEM_TRANSITIONS.get(current, set())
