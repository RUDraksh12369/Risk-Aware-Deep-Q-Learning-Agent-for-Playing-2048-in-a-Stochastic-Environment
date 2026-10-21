"""Per-episode exports and descriptive analysis of saved comparison data."""

import csv
import hashlib
import json
import shutil

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from game.environment import Env2048
from rl.agent import DQNAgent, DQNConfig
from rl.baselines import RandomAgent
from training.analyze import analyze_comparison, trailing_mean
from training.compare import run_comparison
from training.evaluate import evaluate
from training.train import TrainingConfig


def test_episode_records_reproduce_aggregates_without_changing_policy_rng():
    records = []
    metrics = evaluate(RandomAgent(7), episodes=3, seed=1000,
                       max_steps_per_episode=16, episode_records=records)
    assert metrics == evaluate(RandomAgent(7), episodes=3, seed=1000, max_steps_per_episode=16)
    assert [row["seed"] for row in records] == [1000, 1001, 1002]
    assert [row["episode"] for row in records] == [1, 2, 3]
    assert np.mean([row["score"] for row in records]) == metrics["mean_score"]
    assert np.mean([row["steps"] for row in records]) == metrics["mean_episode_length"]
    assert np.mean([row["terminated"] for row in records]) == metrics["game_over_rate"]
    assert np.mean([row["truncated"] for row in records]) == metrics["truncation_rate"]
    assert all(row["terminated"] != row["truncated"] for row in records)


def test_episode_risk_aggregates_use_decision_state_weights(monkeypatch):
    from training import evaluate as evaluation_module

    def short_env(seed):
        env = Env2048(seed=seed)
        original_step = env.step
        count = 0

        def step(action):
            nonlocal count
            state, reward, done, info = original_step(action)
            count += 1
            env.game.done = count >= seed - 999
            return state, reward, env.done, info

        env.step = step
        return env

    monkeypatch.setattr(evaluation_module, "Env2048", short_env)
    records = []
    metrics = evaluate(RandomAgent(7), episodes=3, seed=1000, episode_records=records,
                       high_risk_threshold=0.0)
    assert [row["decision_states"] for row in records] == [1, 2, 3]
    assert metrics["game_over_rate"] == 1 and metrics["truncation_rate"] == 0
    for key in ("mean_board_risk", "mean_empty_cells", "mean_legal_moves"):
        weighted = sum(row[key] * row["decision_states"] for row in records) / 6
        assert metrics[key] == pytest.approx(weighted)
    assert sum(row["high_risk_count"] for row in records) / 6 == metrics["high_risk_rate"]


def test_evaluation_cli_exports_episodes_and_refuses_overwrite(tmp_path, monkeypatch):
    from training import evaluate as evaluation_module

    checkpoint = tmp_path / "agent.pt"
    DQNAgent(DQNConfig(hidden_dims=(8,), seed=7, device="cpu")).save(checkpoint)
    output = tmp_path / "episodes.csv"
    monkeypatch.setattr("sys.argv", ["evaluate", "--checkpoint", str(checkpoint),
                                    "--device", "cpu", "--episodes", "2",
                                    "--max-steps-per-episode", "2", "--episodes-csv", str(output)])
    evaluation_module.main()
    before = output.read_bytes()
    with output.open() as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 2 and records[0]["truncated"] == "True"
    with pytest.raises(SystemExit) as error:
        evaluation_module.main()
    assert error.value.code == 2
    assert output.read_bytes() == before


def test_trailing_mean_is_causal_and_keeps_missing_losses_missing():
    result = trailing_mean(np.array([np.nan, 2.0, 4.0, np.nan, 8.0]), 2)
    np.testing.assert_allclose(result, [np.nan, 2.0, 3.0, np.nan, 8.0], equal_nan=True)
    with pytest.raises(ValueError):
        trailing_mean(np.ones(3), 0)


@pytest.fixture(scope="module")
def comparison_dir(tmp_path_factory):
    output = tmp_path_factory.mktemp("analysis-input") / "comparison"
    run_comparison(output, variants=("random", "dqn", "ra_d3qn"), seeds=(7, 8),
                   training=TrainingConfig(episodes=2, max_steps_per_episode=4,
                                           epsilon_start=0.0, epsilon_end=0.0),
                   dqn=DQNConfig(hidden_dims=(8,), batch_size=2, min_replay_size=2,
                                 replay_capacity=16, device="cpu"),
                   evaluation_episodes=2, evaluation_max_steps=4)
    return output


def test_analysis_creates_report_and_readable_figures_without_modifying_inputs(comparison_dir, tmp_path):
    from matplotlib.image import imread

    originals = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in comparison_dir.rglob("*") if path.is_file()}
    output = tmp_path / "analysis"
    report = analyze_comparison(comparison_dir, output, smoothing_window=2)
    text = report.read_text()
    assert "**completed**" in text
    assert "recorded agent/seed runs: **6**" in text
    assert "not confidence intervals" in text
    assert "Truncated evaluation episodes" in text
    assert "not independent training replicates" in text
    for name in ("evaluation.png", "training.png", "episode_scores.png"):
        pixels = imread(output / name)
        assert pixels.shape[0] >= 500 and pixels.shape[1] >= 800
        assert np.ptp(pixels[:, :, :3]) > 0.5
    for path, digest in originals.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        analyze_comparison(comparison_dir, output)


def test_older_partial_comparisons_are_labeled_and_missing_episode_files_are_not_invented(comparison_dir, tmp_path):
    copied = tmp_path / "old"
    shutil.copytree(comparison_dir, copied)
    manifest = json.loads((copied / "manifest.json").read_text())
    manifest["status"] = "failed"
    (copied / "manifest.json").write_text(json.dumps(manifest))
    with (copied / "results.csv").open() as stream:
        records = list(csv.DictReader(stream))[:1]
    records[0].pop("evaluation_history")
    with (copied / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = analyze_comparison(copied)
    text = report.read_text()
    assert "Partial comparison" in text
    assert "1 runs lack episode exports" in text
    assert "SD not estimated" in text
    assert not (report.parent / "episode_scores.png").exists()
    assert not (report.parent / "training.png").exists()  # First row is random.


@pytest.mark.parametrize("corruption", ["duplicate", "nonfinite", "path_escape", "wrong_episode_count"])
def test_invalid_analysis_data_is_rejected_before_creating_output(comparison_dir, tmp_path, corruption):
    copied = tmp_path / "bad"
    shutil.copytree(comparison_dir, copied)
    with (copied / "results.csv").open() as stream:
        records = list(csv.DictReader(stream))
    if corruption == "duplicate":
        records.append(records[0].copy())
    elif corruption == "nonfinite":
        records[0]["mean_score"] = "nan"
    elif corruption == "path_escape":
        records[0]["evaluation_history"] = "../outside.csv"
    else:
        records[0]["episodes"] = "200"
    with (copied / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    output = tmp_path / "analysis"
    with pytest.raises(ValueError):
        analyze_comparison(copied, output)
    assert not output.exists()
