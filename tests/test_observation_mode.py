from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / "deploy" / name).read_text(encoding="utf-8")


def test_core_tick_timer_runs_every_five_minutes():
    timer = _read("agent-core-tick.timer")
    assert "OnUnitActiveSec=5min" in timer
    assert "Persistent=true" in timer


def test_lab_tick_timer_runs_only_gate_command_periodically():
    service = _read("agent-core-lab-tick.service")
    timer = _read("agent-core-lab-tick.timer")
    assert "agentctl lab tick-if-enabled --notify" in service
    assert "OnUnitActiveSec=30min" in timer
    assert "Persistent=true" in timer


def test_daily_and_activity_summary_timers_are_separate():
    activity = _read("agent-core-summary.service")
    daily = _read("agent-core-daily-summary.service")
    daily_timer = _read("agent-core-daily-summary.timer")
    assert "agentctl notify activity-summary" in activity
    assert "agentctl notify daily-summary" in daily
    assert "OnCalendar=*-*-* 09:00:00" in daily_timer


def test_deploy_services_are_user_systemd_units():
    services = [
        "agent-core-discord.service",
        "agent-core-tick.service",
        "agent-core-lab-tick.service",
        "agent-core-summary.service",
        "agent-core-daily-summary.service",
    ]
    for name in services:
        text = _read(name)
        assert "User=ubuntu" not in text
    assert "WantedBy=default.target" in _read("agent-core-discord.service")
