from agent.config.defaults import db_path, env_path, state_path, workspace_root
from agent.core.database import init_db


def test_pytest_runtime_uses_isolated_paths(tmp_path):
    init_db()

    assert db_path() == tmp_path / "agent.db"
    assert state_path() == tmp_path / "state.json"
    assert env_path() == tmp_path / ".env"
    assert workspace_root() == tmp_path / "workspace"
