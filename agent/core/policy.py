from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal
import json
import re

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
RiskLevel = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True)
class ActionProposal:
    action_type: str
    description: str
    payload: dict[str, Any]
    risk_level: str
    requires_approval: bool
    denied_reason: str | None = None

    @property
    def denied(self) -> bool:
        return self.denied_reason is not None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["denied"] = self.denied
        return result


@dataclass(frozen=True)
class PolicyDecision:
    input_text: str
    normalized_text: str
    action_type: str
    risk_level: RiskLevel
    requires_approval: bool
    denied: bool
    reason: str
    matched_rules: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyRule:
    name: str
    pattern: re.Pattern[str]
    risk_level: RiskLevel
    requires_approval: bool
    denied_reason: str | None = None

    def matches(self, text: str) -> bool:
        return self.pattern.search(text) is not None


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


class PolicyEngine:
    deny_rules: tuple[PolicyRule, ...] = (
        PolicyRule("root_delete", _rx(r"(?:^|[;&|\s])rm\s+(?:--\s+)?-[^\s]*[rR][^\s]*[fF][^\s]*(?:\s+--)?\s+/(?:\*|\.{1,2}(?:/)?|\s|$)"), "critical", True, "root_delete_denied"),
        PolicyRule("ssh_dir", _rx(r"(?:~|/home/[^\s;&|]+|/root)/\.ssh(?:/|\s|$)"), "critical", True, "ssh_key_access_denied"),
        PolicyRule("private_key", _rx(r"\b(id_rsa|id_ed25519|authorized_keys)\b"), "critical", True, "ssh_key_access_denied"),
        PolicyRule("env_file", _rx(r"(?:^|[\s;&|/])\.env(?:\s|$)"), "critical", True, "env_access_denied"),
        PolicyRule("secret_text", _rx(r"\b(secret|api[_-]?key|authorization:\s*bearer|token=|token:|token\.(?:txt|json|env|key)|secret\.(?:txt|json|env|key))"), "critical", True, "secret_access_denied"),
        PolicyRule("external_secret_send", _rx(r"\b(external secret|send secret|upload .*secret)\b"), "critical", True, "external_secret_send_denied"),
    )
    approval_rules: tuple[PolicyRule, ...] = (
        PolicyRule("sudo", _rx(r"(?:^|[;&|\s])sudo(?:\s|$)"), "critical", True),
        PolicyRule("etc_path", _rx(r"(?:^|[\s;&|])/(?:etc)(?:/|\s|$)"), "high", True),
        PolicyRule("systemctl_write", _rx(r"\bsystemctl\s+(enable|disable|restart|start|stop|mask|unmask)\b"), "high", True),
        PolicyRule("apt_write", _rx(r"\bapt(?:-get)?\s+(install|remove|purge|upgrade|dist-upgrade|full-upgrade|autoremove)\b"), "high", True),
        PolicyRule("file_write", _rx(r"(?:^|[;&|\s])(rm|mv|chmod|chown|truncate|dd)\b"), "high", True),
        PolicyRule("network_fetch", _rx(r"(?:^|[;&|\s])(curl|wget|scp|rsync)\b"), "medium", True),
    )
    read_only_rules: tuple[PolicyRule, ...] = (
        PolicyRule("df", _rx(r"^\s*df\s+-h(?:\s+/)?\s*$"), "medium", False),
        PolicyRule("free", _rx(r"^\s*free\s+-h\s*$"), "medium", False),
        PolicyRule("swapon", _rx(r"^\s*swapon\s+--show\s*$"), "medium", False),
        PolicyRule("uptime", _rx(r"^\s*uptime\s*$"), "medium", False),
        PolicyRule("journal_usage", _rx(r"^\s*journalctl\s+--disk-usage\s*$"), "medium", False),
        PolicyRule("systemctl_failed", _rx(r"^\s*systemctl\s+--failed(?:\s+--no-pager)?\s*$"), "medium", False),
        PolicyRule("git_status", _rx(r"^\s*git\s+(status\s+--short|log\s+--oneline\s+-5)\s*$"), "medium", False),
    )
    internal_actions = {
        "memory_add", "memory_search", "event_log", "state_read", "state_update",
        "goal_create", "goal_list", "decision_record", "renderer_fallback", "reflection_create",
        "skill_update", "discord_route", "eval_run",
    }

    def classify_text(self, text: str, action_type: str = "shell_text") -> ActionProposal:
        decision = self.classify_decision(text, action_type=action_type)
        return ActionProposal(
            action_type=decision.action_type,
            description=text,
            payload={"text": text, "normalized_text": decision.normalized_text, "matched_rules": decision.matched_rules},
            risk_level=decision.risk_level,
            requires_approval=decision.requires_approval,
            denied_reason=decision.reason if decision.denied else None,
        )

    def classify_decision(self, text: str, action_type: str = "shell_text") -> PolicyDecision:
        normalized = self._normalize_text(text)
        matched: list[str] = []
        risk_level: RiskLevel = "low"
        requires_approval = False
        denied_reason: str | None = None

        for rule in self.deny_rules:
            if rule.matches(normalized):
                matched.append(rule.name)
                risk_level = self._max_risk(risk_level, rule.risk_level)  # type: ignore[assignment]
                requires_approval = requires_approval or rule.requires_approval
                denied_reason = denied_reason or rule.denied_reason

        for rule in self.approval_rules:
            if rule.matches(normalized):
                matched.append(rule.name)
                risk_level = self._max_risk(risk_level, rule.risk_level)  # type: ignore[assignment]
                requires_approval = requires_approval or rule.requires_approval

        if not matched:
            for rule in self.read_only_rules:
                if rule.matches(normalized):
                    matched.append(rule.name)
                    risk_level = self._max_risk(risk_level, rule.risk_level)  # type: ignore[assignment]
                    requires_approval = False
                    break

        return PolicyDecision(
            input_text=text,
            normalized_text=normalized,
            action_type=action_type,
            risk_level=risk_level,
            requires_approval=requires_approval,
            denied=denied_reason is not None,
            reason=denied_reason or ("approval_required" if requires_approval else "allowed"),
            matched_rules=matched,
        )

    def record_decision(self, decision: PolicyDecision) -> int:
        init_db()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO policy_decisions (
                    created_at, input_text, normalized_text, action_type, risk_level,
                    requires_approval, denied, reason, matched_rules_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_kst(), decision.input_text, decision.normalized_text, decision.action_type,
                    decision.risk_level, 1 if decision.requires_approval else 0,
                    1 if decision.denied else 0, decision.reason,
                    json.dumps(decision.matched_rules, ensure_ascii=False),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def classify_action(self, action_type: str, description: str, payload: dict[str, Any] | None = None) -> ActionProposal:
        payload = payload or {}
        if action_type in self.internal_actions:
            return ActionProposal(action_type, description, payload, "low", False)
        return self.classify_text(f"{description} {payload}", action_type=action_type)

    def _normalize_text(self, text: str) -> str:
        text = text.strip().lower().replace("'", " ").replace('"', " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*([;&|])\s*", r" \1 ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _max_risk(self, current: str, candidate: str) -> str:
        return candidate if RISK_ORDER[candidate] > RISK_ORDER[current] else current


def classify_text_action(text: str) -> dict[str, object]:
    proposal = PolicyEngine().classify_text(text)
    return {
        "risk_level": proposal.risk_level,
        "requires_approval": proposal.requires_approval,
        "blocked_tokens": proposal.payload.get("matched_rules", []),
        "denied_reason": proposal.denied_reason,
        "denied": proposal.denied,
    }
