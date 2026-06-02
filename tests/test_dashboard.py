import json
import threading
import urllib.request

from agent.cli.agentctl import main
from agent.core.approvals import ApprovalStore
from agent.core.database import init_db
from agent.core.policy import ActionProposal
from agent.dashboard.server import DashboardRequestHandler
from agent.dashboard.snapshot import dashboard_snapshot
from agent.tools.action_log import create_action_run, finish_action_run
from http.server import ThreadingHTTPServer


def _json_tree(value):
    return json.dumps(value, ensure_ascii=False)


def test_dashboard_snapshot_redacts_sensitive_raw_fields(capsys):
    init_db()
    action_id = create_action_run(
        goal_id=None,
        action_type="local_shell",
        command=["echo", "token=abc123"],
        cwd="/home/ubuntu/agent_core",
        profile="full_device_lab",
        risk_level="medium",
        status="running",
        result_summary="token check",
    )
    finish_action_run(
        action_id,
        status="completed",
        returncode=0,
        stdout="secret stdout token=abc123",
        stderr="password stderr",
        after_snapshot={},
        result_summary="completed",
    )
    ApprovalStore().create_approval(
        ActionProposal(
            action_type="shell_text",
            description="sudo 작업 확인",
            payload={"token": "abc123", "command": "cat .env"},
            risk_level="high",
            requires_approval=True,
        )
    )

    snapshot = dashboard_snapshot(limit=5)
    tree = _json_tree(snapshot)

    assert snapshot["kind"] == "agent_core_dashboard_snapshot"
    assert snapshot["safety"]["raw_outputs_exposed"] is False
    assert snapshot["safety"]["approval_payloads_exposed"] is False
    assert "proposed_payload_json" not in tree
    assert "secret stdout" not in tree
    assert "password stderr" not in tree
    assert "abc123" not in tree
    assert snapshot["actions"]["items"][0]["has_output"] is True

    assert main(["dashboard", "snapshot", "--limit", "5"]) == 0
    cli_snapshot = json.loads(capsys.readouterr().out)
    assert cli_snapshot["kind"] == "agent_core_dashboard_snapshot"


def test_dashboard_http_server_serves_health_and_snapshot():
    init_db()
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
            health = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(f"{base}/api/snapshot?limit=3", timeout=5) as response:
            snapshot = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(base, timeout=5) as response:
            index = response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert health["ok"] is True
    assert snapshot["processes"]["items"] == []
    assert "Control Room" in index
