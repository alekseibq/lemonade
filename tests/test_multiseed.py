from __future__ import annotations

import pytest

pytest.importorskip("stable_baselines3")

from lemonade.training import train_multiple_seeds


def test_multiseed_report_uses_same_held_out_scenarios(tmp_path) -> None:
    report = train_multiple_seeds(tmp_path, [1, 2], 128, [50, 51])
    assert report["training_runs"] == 2
    assert report["evaluation_seeds"] == [50, 51]
    assert (tmp_path / "comparison.json").is_file()
    assert all(run["evaluation"]["random"]["episodes"] == 2 for run in report["per_run"])
