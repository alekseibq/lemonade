from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("stable_baselines3")

from lemonade import api
from lemonade.training import TrainingConfig, train_and_evaluate


def test_replay_returns_one_step_without_future_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_and_evaluate(
        tmp_path / "smoke",
        TrainingConfig(total_timesteps=128, n_steps=64, batch_size=32),
        evaluation_seeds=[10],
    )
    monkeypatch.setattr(api, "ARTIFACTS_DIR", tmp_path)
    api.replays.clear()
    client = TestClient(api.app)
    assert client.get("/replay-models").json() == {"models": ["smoke"]}
    created = client.post("/replays", json={"model_id": "smoke", "seed": 42})
    assert created.status_code == 201
    first = client.post(f"/replays/{created.json()['replay_id']}/next")
    payload = first.json()
    assert set(payload) == {"step", "observation", "action", "result", "terminated"}
    assert payload["step"] == 1
    assert "demand_noise" not in payload
    assert client.post(f"/replays/{created.json()['replay_id']}/next").status_code == 200
