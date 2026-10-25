from __future__ import annotations

import pytest

pytest.importorskip("stable_baselines3")

from lemonade.training import TrainingConfig, load_and_evaluate, train_and_evaluate


def test_ppo_smoke_saves_loadable_artifact_and_report(tmp_path) -> None:
    report = train_and_evaluate(
        tmp_path / "smoke",
        training=TrainingConfig(total_timesteps=128, n_steps=64, batch_size=32),
        evaluation_seeds=[100, 101],
    )
    assert (tmp_path / "smoke" / "model.zip").is_file()
    assert (tmp_path / "smoke" / "report.json").is_file()
    assert report["evaluation"]["trained"]["episodes"] == 2
    loaded = load_and_evaluate(tmp_path / "smoke" / "model.zip", [100, 101])
    assert loaded.episodes == 2
