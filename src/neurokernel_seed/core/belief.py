from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SLOT_VALUE_VOCAB = (
    "unknown",
    "digit.1",
    "digit.2",
    "digit.3",
    "digit.4",
    "tool.search",
    "tool.read",
    "tool.summarize",
    "color.red",
    "color.blue",
    "color.green",
)


@dataclass(frozen=True)
class SlotBelief:
    index: int
    value_key: str
    known: bool
    revealed: bool
    confidence: float

    @property
    def unknown(self) -> bool:
        return not self.known


@dataclass(frozen=True)
class ObservationBelief:
    kind: str
    slot: int
    value_key: str


@dataclass(frozen=True)
class BeliefState:
    env_family: str
    visibility_mode: str
    slots: tuple[SlotBelief, ...]
    current_index: int
    sequence_length: int
    remaining_steps: int
    progress_fraction: float
    current_slot_known: bool
    current_required_value: str
    last_observation: ObservationBelief
    pending_information_need: bool
    executable_now: bool
    can_finish: bool

    @property
    def known_slots(self) -> tuple[int, ...]:
        return tuple(slot.index for slot in self.slots if slot.known)

    @property
    def unknown_slots(self) -> tuple[int, ...]:
        return tuple(slot.index for slot in self.slots if slot.unknown)

    @property
    def revealed_slots(self) -> tuple[int, ...]:
        return tuple(slot.index for slot in self.slots if slot.revealed)


def belief_from_facts(env_name: str, facts: dict[str, Any]) -> BeliefState:
    family = env_name.split(".", 1)[0]
    visibility_mode = str(facts.get("visibility_mode") or "visible")
    slots = _slots_for_family(family, facts)
    sequence_length = int(facts.get("sequence_length") or len(slots) or 1)
    current_index = int(facts.get("current_index", facts.get("progress", facts.get("stage", 0))) or 0)
    max_steps = int(facts.get("max_steps") or 1)
    steps = int(facts.get("steps") or 0)
    remaining_steps = int(facts.get("remaining_steps", max_steps - steps))
    progress_fraction = _clip01(float(facts.get("progress_fraction", current_index / max(1, sequence_length))))
    current_slot_known = _current_slot_known(facts, slots, current_index)
    current_required_value = slots[current_index].value_key if current_slot_known and 0 <= current_index < len(slots) else "unknown"
    last_observation = _last_observation(facts)
    pending_information_need = current_index < sequence_length and not current_slot_known
    executable_now = current_index < sequence_length and current_slot_known
    can_finish = bool(facts.get("can_finish", facts.get("can_summarize", False)))
    return BeliefState(
        env_family=family,
        visibility_mode=visibility_mode,
        slots=tuple(slots),
        current_index=current_index,
        sequence_length=sequence_length,
        remaining_steps=max(0, remaining_steps),
        progress_fraction=progress_fraction,
        current_slot_known=current_slot_known,
        current_required_value=current_required_value,
        last_observation=last_observation,
        pending_information_need=pending_information_need,
        executable_now=executable_now,
        can_finish=can_finish,
    )


def value_key(value: Any, family: str) -> str:
    if value in (None, "", 0, "unknown"):
        return "unknown"
    if family == "lock":
        try:
            digit = int(value)
        except (TypeError, ValueError):
            return "unknown"
        return f"digit.{digit}" if 1 <= digit <= 4 else "unknown"
    if family == "tool":
        text = str(value)
        return f"tool.{text}" if f"tool.{text}" in SLOT_VALUE_VOCAB else "unknown"
    if family == "maze":
        text = str(value)
        return f"color.{text}" if f"color.{text}" in SLOT_VALUE_VOCAB else "unknown"
    return "unknown"


def _slots_for_family(family: str, facts: dict[str, Any]) -> list[SlotBelief]:
    if family == "lock":
        values = list(facts.get("visible_code") or facts.get("code") or [])
    elif family == "tool":
        values = list(facts.get("visible_sequence") or facts.get("sequence") or [])
    elif family == "maze":
        doors = list(facts.get("doors") or [])
        target = facts.get("visible_target_color", facts.get("target_color"))
        values = doors or ([target] if target not in (None, "") else [])
    else:
        values = []
    if family == "maze":
        length = len(values)
    else:
        length = int(facts.get("sequence_length") or len(values))
    known_mask = _mask(facts.get("known_mask"), length, default=True)
    unknown_mask = _mask(facts.get("unknown_mask"), length, default=False)
    slots: list[SlotBelief] = []
    for index in range(length):
        raw_value = values[index] if index < len(values) else None
        known = bool(known_mask[index]) and not bool(unknown_mask[index]) and value_key(raw_value, family) != "unknown"
        revealed = known or bool(known_mask[index])
        slots.append(SlotBelief(index=index, value_key=value_key(raw_value, family), known=known, revealed=revealed, confidence=1.0 if known else 0.0))
    return slots


def _current_slot_known(facts: dict[str, Any], slots: list[SlotBelief], current_index: int) -> bool:
    if "current_slot_known" in facts:
        return bool(facts["current_slot_known"])
    return current_index >= len(slots) or (0 <= current_index < len(slots) and slots[current_index].known)


def _last_observation(facts: dict[str, Any]) -> ObservationBelief:
    raw = facts.get("last_observation") if isinstance(facts.get("last_observation"), dict) else {}
    kind = str(facts.get("last_observation_type", raw.get("type", "none")) or "none")
    slot = int(facts.get("last_observation_slot", raw.get("slot", -1)) or -1)
    if "target_color" in facts or "doors" in facts:
        family = "maze"
    elif "code" in facts or isinstance(raw.get("value"), int):
        family = "lock"
    else:
        family = "tool"
    return ObservationBelief(kind=kind, slot=slot, value_key=value_key(facts.get("last_observation_value", raw.get("value")), family))


def _mask(value: Any, length: int, *, default: bool) -> list[bool]:
    if isinstance(value, (list, tuple)) and len(value) == length:
        return [bool(item) for item in value]
    return [default] * length


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
