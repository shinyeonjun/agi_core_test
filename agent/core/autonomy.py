from __future__ import annotations

from typing import Any, Literal

from agent.core.state import load_state, save_state

AutonomyProfile = Literal["safe", "workspace", "full_device_lab"]
VALID_PROFILES = {"safe", "workspace", "full_device_lab"}

PROFILE_SETTINGS: dict[str, dict[str, Any]] = {
    "safe": {
        "autonomy_profile": "safe",
        "full_device_lab_enabled": False,
        "external_network_actions_allowed": False,
        "os_mutation_allowed": False,
        "self_modification_allowed": "proposal_only",
    },
    "workspace": {
        "autonomy_profile": "workspace",
        "full_device_lab_enabled": False,
        "external_network_actions_allowed": False,
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
    ]}


def set_autonomy_profile(profile: str) -> dict[str, Any]:
    if profile not in VALID_PROFILES:
        raise ValueError(f"invalid autonomy profile: {profile}")
    state = load_state()
    state.update(PROFILE_SETTINGS[profile])
    save_state(state)
    return get_autonomy_state()


def current_profile() -> str:
    return str(load_state().get("autonomy_profile", "safe"))
