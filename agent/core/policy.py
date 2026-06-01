from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass(frozen=True)
class ActionProposal:
    action_type: str
    description: str
    payload: dict[str, Any]
    risk_level: str
    requires_approval: bool
    denied_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyRule:
    token: str
    risk_level: str
    requires_approval: bool
    denied_reason: str | None = None


class PolicyEngine:
    deny_rules: tuple[PolicyRule, ...] = (
        PolicyRule("rm -rf /", "critical", True, "root_delete_denied"),
        PolicyRule("~/.ssh", "critical", True, "ssh_key_access_denied"),
        PolicyRule("/home/ubuntu/.ssh", "critical", True, "ssh_key_access_denied"),
        PolicyRule("id_rsa", "critical", True, "ssh_key_access_denied"),
        PolicyRule(".env", "critical", True, "env_access_denied"),
        PolicyRule("secret", "critical", True, "secret_access_denied"),
        PolicyRule("token", "critical", True, "secret_access_denied"),
    )

    approval_rules: tuple[PolicyRule, ...] = (
        PolicyRule("sudo", "critical", True),
        PolicyRule("/etc/", "high", True),
        PolicyRule("/etc ", "high", True),
        PolicyRule("systemctl enable", "high", True),
        PolicyRule("systemctl disable", "high", True),
        PolicyRule("systemctl restart", "high", True),
        PolicyRule("apt install", "high", True),
        PolicyRule("apt remove", "high", True),
        PolicyRule("apt purge", "high", True),
        PolicyRule("apt upgrade", "high", True),
        PolicyRule("apt-get install", "high", True),
        PolicyRule("apt-get remove", "high", True),
        PolicyRule("apt-get purge", "high", True),
        PolicyRule("apt-get upgrade", "high", True),
        PolicyRule("rm ", "high", True),
        PolicyRule("mv ", "high", True),
        PolicyRule("chmod ", "high", True),
        PolicyRule("chown ", "high", True),
        PolicyRule("curl ", "medium", True),
        PolicyRule("wget ", "medium", True),
        PolicyRule("scp ", "medium", True),
        PolicyRule("rsync ", "medium", True),
    )

    read_only_tokens = (
        "df ",
        "free ",
        "uptime",
        "uname",
        "systemctl status",
        "journalctl --disk-usage",
        "ls ",
        "cat /proc/",
        "python -c",
    )

    internal_actions = {
        "memory_add",
        "memory_search",
        "event_log",
        "state_read",
        "state_update",
        "goal_create",
        "goal_list",
        "decision_record",
        "renderer_fallback",
    }

    def classify_text(self, text: str, action_type: str = "shell_text") -> ActionProposal:
        normalized = self._normalize_text(text)
        matched: list[str] = []
        risk_level = "low"
        requires_approval = False
        denied_reason: str | None = None

        for rule in self.deny_rules:
            if rule.token.lower() in normalized:
                matched.append(rule.token)
                risk_level = self._max_risk(risk_level, rule.risk_level)
                requires_approval = requires_approval or rule.requires_approval
                denied_reason = denied_reason or rule.denied_reason

        for rule in self.approval_rules:
            if rule.token.lower() in normalized:
                matched.append(rule.token)
                risk_level = self._max_risk(risk_level, rule.risk_level)
                requires_approval = requires_approval or rule.requires_approval

        if not matched and any(token in normalized for token in self.read_only_tokens):
            risk_level = "medium"
            requires_approval = False

        if "external secret" in normalized or "send secret" in normalized:
            risk_level = "critical"
            requires_approval = True
            denied_reason = denied_reason or "external_secret_send_denied"
            matched.append("external_secret_send")

        return ActionProposal(
            action_type=action_type,
            description=text,
            payload={"text": text, "matched_rules": matched},
            risk_level=risk_level,
            requires_approval=requires_approval,
            denied_reason=denied_reason,
        )

    def classify_action(
        self,
        action_type: str,
        description: str,
        payload: dict[str, Any] | None = None,
    ) -> ActionProposal:
        payload = payload or {}
        if action_type in self.internal_actions:
            return ActionProposal(action_type, description, payload, "low", False)
        if action_type in {"system_read", "file_read"}:
            text = f"{description} {payload}"
            proposal = self.classify_text(text, action_type=action_type)
            if proposal.risk_level == "low":
                return ActionProposal(action_type, description, payload, "medium", False)
            return proposal
        return self.classify_text(f"{description} {payload}", action_type=action_type)

    def _normalize_text(self, text: str) -> str:
        return " " + re.sub(r"\s+", " ", text.strip().lower()) + " "

    def _max_risk(self, current: str, candidate: str) -> str:
        return candidate if RISK_ORDER[candidate] > RISK_ORDER[current] else current


def classify_text_action(text: str) -> dict[str, object]:
    proposal = PolicyEngine().classify_text(text)
    return {
        "risk_level": proposal.risk_level,
        "requires_approval": proposal.requires_approval,
        "blocked_tokens": proposal.payload.get("matched_rules", []),
        "denied_reason": proposal.denied_reason,
    }
