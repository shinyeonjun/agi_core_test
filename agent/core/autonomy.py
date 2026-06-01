from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from agent.core.state import load_state, save_state

AutonomyProfile = Literal["safe", "workspace", "full_device_lab"]
VALID_PROFILES = {"safe", "workspace", "full_device_lab"}

PROFILE_SETTINGS: dict[str, dict[str, Any]] = {
    "safe": {
        "autonomy_profile": "safe",
        "full_device_lab_enabled": False,
        "external_network_actions_allowed": False,
        "catastrophic_local_destruction_allowed": False,
        "catastrophic_local_destruction_armed_until": None,
        "os_mutation_allowed": False,
        "self_modification_allowed": "proposal_only",
    },
    "workspace": {
        "autonomy_profile": "workspace",
        "full_device_lab_enabled": False,
        "external_network_actions_allowed": False,
        "catastrophic_local_destruction_allowed": False,
        "catastrophic_local_destruction_armed_until": None,
        "os_mutation_allowed": False,
        "self_modification_allowed": "proposal_only",
    },
    "full_device_lab": {
        "autonomy_profile": "full_device_lab",
        "full_device_lab_enabled": True,
        "external_network_actions_allowed": False,
        "os_mutation_allowed": True,
        "self_modification_allowed": "branch_or_proposal",
    },
}


def get_autonomy_state() -> dict[str, Any]:
    state = load_state()
    return {key: state.get(key) for key in [
        "autonomy_profile",
        "full_device_lab_enabled",
        "external_network_actions_allowed",
        "os_mutation_allowed",
        "self_modification_allowed",
        "catastrophic_local_destruction_allowed",
        "catastrophic_local_destruction_armed_until",
    ]}


def set_autonomy_profile(profile: str) -> dict[str, Any]:
    if profile not in VALID_PROFILES:
        raise ValueError(f"invalid autonomy profile: {profile}")
    state = load_state()
    state.update(PROFILE_SETTINGS[profile])
    save_state(state)
    return get_autonomy_state()


def arm_catastrophic_destruction(ttl_seconds: int = 300) -> dict[str, Any]:
    if ttl_seconds <= 0 or ttl_seconds > 3600:
        raise ValueError("ttl_seconds must be between 1 and 3600")
    state = load_state()
    if state.get("autonomy_profile") != "full_device_lab":
        raise ValueError("catastrophic destruction arm requires full_device_lab profile")
    expires_at = (datetime.now().astimezone() + timedelta(seconds=ttl_seconds)).isoformat(timespec="seconds")
    state["catastrophic_local_destruction_allowed"] = True
    state["catastrophic_local_destruction_armed_until"] = expires_at
    save_state(state)
    return get_autonomy_state()


def disarm_catastrophic_destruction() -> dict[str, Any]:
    state = load_state()
    state["catastrophic_local_destruction_allowed"] = False
    state["catastrophic_local_destruction_armed_until"] = None
    save_state(state)
    return get_autonomy_state()


def is_catastrophic_destruction_armed() -> bool:
    state = load_state()
    if state.get("autonomy_profile") != "full_device_lab":
        if state.get("catastrophic_local_destruction_allowed"):
            disarm_catastrophic_destruction()
        return False
    if not state.get("catastrophic_local_destruction_allowed"):
        return False
    armed_until = state.get("catastrophic_local_destruction_armed_until")
    if not armed_until:
        disarm_catastrophic_destruction()
        return False
    try:
        expires_at = datetime.fromisoformat(str(armed_until))
    except ValueError:
        disarm_catastrophic_destruction()
        return False
    if datetime.now(expires_at.tzinfo).astimezone(expires_at.tzinfo) >= expires_at:
        disarm_catastrophic_destruction()
        return False
    return True


def current_profile() -> str:
    return str(load_state().get("autonomy_profile", "safe"))
