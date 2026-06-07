import json
import os
import sqlite3
import subprocess
import sys


def test_cli_lists_test_envs():
    result = subprocess.run(
        [sys.executable, "-m", "neurokernel_seed.cli", "list-envs", "--split", "test"],
        check=True,
        text=True,
        capture_output=True,
        env=os.environ | {"PYTHONPATH": "src"},
    )
    assert "lock.test" in result.stdout
    assert "tool.test" in result.stdout
    assert "maze.test" in result.stdout


def test_cli_run_suite_logs_all_test_envs_to_one_db(tmp_path):
    db = tmp_path / "suite.db"
    result = subprocess.run(
        [sys.executable, "-m", "neurokernel_seed.cli", "run-suite", "--split", "test", "--episodes", "1", "--db", str(db)],
        check=True,
        text=True,
        capture_output=True,
        env=os.environ | {"PYTHONPATH": "src"},
    )
    payload = json.loads(result.stdout)
    assert set(payload) == {"lock.test", "maze.test", "tool.test"}
    conn = sqlite3.connect(db)
    episodes = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    envs = {row[0] for row in conn.execute("SELECT DISTINCT env_name FROM episodes")}
    metadata = [json.loads(row[0]) for row in conn.execute("SELECT env_metadata_json FROM episodes")]
    assert episodes == 3
    assert envs == {"lock.test", "maze.test", "tool.test"}
    assert all(item.get("split") == "test" for item in metadata)
