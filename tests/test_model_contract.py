import json
import subprocess
import sys
from pathlib import Path

from neurokernel_seed.model.dataset import load_feature_manifest, load_feature_rows
from neurokernel_seed.replay.dataset import canonical_action_key, validate_flat_features


def test_model_feature_loader_contract(tmp_path: Path):
    features = tmp_path / "features.jsonl"
    manifest = tmp_path / "features.jsonl.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": 1,
                "schema_version": "neurokernel-flat-transition-v1",
                "replay_schema_version": "neurokernel-replay-v2",
                "envs": ["lock.test"],
                "splits": ["test"],
                "action_vocab": ["press(digit=1)"],
                "max_state_dim": 3,
                "state_dim_by_env": {"lock.test": 3},
                "input_dim": 5,
                "target_dim": 5,
                "input_layout": {"env_onehot": [0, 1], "state_padded": [1, 4], "action_onehot": [4, 5]},
                "target_layout": {"next_state_padded": [0, 3], "reward": 3, "done": 4},
            }
        ),
        encoding="utf-8",
    )
    features.write_text(
        json.dumps(
            {
                "schema_version": "neurokernel-flat-transition-v1",
                "row_index": 0,
                "episode_id": "e1",
                "step_index": 0,
                "env_name": "lock.test",
                "split": "test",
                "action_key": "press(digit=1)",
                "input_vector": [1, 0, 0, 0, 1],
                "target_vector": [0.5, 0.1, 0, 1, 0],
                "target_mask": [1, 1, 1, 1, 1],
                "source": {"prev_state_id": "a", "next_state_id": "b"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checked = validate_flat_features(features)
    assert checked["rows"] == 1
    assert load_feature_manifest(features)["input_dim"] == 5
    rows = load_feature_rows(features, "test")
    assert rows[0].action_key == "press(digit=1)"


def test_canonical_action_key_is_stable():
    assert canonical_action_key({"name": "move", "params": {"door": "green"}}) == 'move(door="green")'
    assert canonical_action_key({"name": "reset", "params": {}}) == "reset"


def test_model_cli_commands_are_registered():
    result = subprocess.run([sys.executable, "-m", "neurokernel_seed.cli", "train-world-model", "--help"], check=True, text=True, capture_output=True)
    assert "--features" in result.stdout
    result = subprocess.run([sys.executable, "-m", "neurokernel_seed.cli", "eval-learned-gate", "--help"], check=True, text=True, capture_output=True)
    assert "--model" in result.stdout
    result = subprocess.run([sys.executable, "-m", "neurokernel_seed.cli", "collect-counterfactual-dataset", "--help"], check=True, text=True, capture_output=True)
    assert "--counterfactual-out" in result.stdout
    result = subprocess.run([sys.executable, "-m", "neurokernel_seed.cli", "eval-action-ranking", "--help"], check=True, text=True, capture_output=True)
    assert "--checkpoint" in result.stdout
    result = subprocess.run([sys.executable, "-m", "neurokernel_seed.cli", "export-world-model-onnx", "--help"], check=True, text=True, capture_output=True)
    assert "--static-batch" in result.stdout
    result = subprocess.run([sys.executable, "-m", "neurokernel_seed.cli", "benchmark-runtime", "--help"], check=True, text=True, capture_output=True)
    assert "--backend" in result.stdout
    result = subprocess.run([sys.executable, "-m", "neurokernel_seed.cli", "harness-create-task", "--help"], check=True, text=True, capture_output=True)
    assert "--task-json" in result.stdout
    result = subprocess.run([sys.executable, "-m", "neurokernel_seed.cli", "serve-core-api", "--help"], check=True, text=True, capture_output=True)
    assert "--port" in result.stdout
    result = subprocess.run([sys.executable, "-m", "neurokernel_seed.cli", "serve-discord-bot", "--help"], check=True, text=True, capture_output=True)
    assert "--channel-id" in result.stdout
    assert "--allowed-user-id" in result.stdout
    assert "--auto-do-low-risk" in result.stdout
    result = subprocess.run([sys.executable, "-m", "neurokernel_seed.cli", "language-to-core", "--help"], check=True, text=True, capture_output=True)
    assert "--no-codex" not in result.stdout
    result = subprocess.run([sys.executable, "-m", "neurokernel_seed.cli", "language-to-human", "--help"], check=True, text=True, capture_output=True)
    assert "--core-result-json" in result.stdout
    assert "--no-codex" not in result.stdout
