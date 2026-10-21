"""Reference policies and reproducible, isolated baseline comparisons."""

import csv
from dataclasses import asdict
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from game.environment import Env2048, encode_board
from game.game import Action
from rl.agent import DQNAgent, DQNConfig
from rl.baselines import GreedyAgent, RandomAgent
from rl.risk_aware import RiskAwareConfig, RiskAwareDQNAgent
from training.compare import VARIANTS, create_agent, run_comparison, summarize_results
from training.evaluate import evaluate
from training.train import TrainingConfig


def tiny_config():
    return DQNConfig(hidden_dims=(8,), batch_size=2, min_replay_size=2,
                     replay_capacity=16, seed=7, device="cpu")


@pytest.mark.parametrize("variant, double, dueling", [
    ("dqn", False, False), ("double_dqn", True, False), ("dueling_dqn", False, True),
    ("dueling_double_dqn", True, True), ("ra_d3qn", True, True),
])
def test_variant_flags_are_explicit_and_base_config_is_unchanged(variant, double, dueling):
    config = tiny_config()
    original = asdict(config)
    agent = create_agent(variant, config, RiskAwareConfig(0.7))
    assert agent.config.double_dqn is double
    assert agent.config.dueling is dueling
    assert agent.online_net.dueling is dueling
    assert asdict(config) == original
    assert agent.learn_steps == 0 and len(agent.replay) == 0
    assert isinstance(agent, RiskAwareDQNAgent) == (variant == "ra_d3qn")
    if variant == "ra_d3qn":
        assert agent.risk_config.risk_coefficient == 0.7


def test_random_policy_is_seeded_and_only_uses_legal_actions():
    mask = np.array([False, True, False, True])
    first, second = RandomAgent(42), RandomAgent(42)
    a = [first.select_action(np.zeros(16), mask) for _ in range(50)]
    b = [second.select_action(np.zeros(16), mask) for _ in range(50)]
    assert a == b
    assert set(a) == {1, 3}


def test_greedy_maximizes_merge_score_and_preserves_board_and_rng():
    env = Env2048(seed=42)
    env.game.board = np.array([[2, 2, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    original = env.game.clone()
    # LEFT and RIGHT both gain 4; UP/DOWN gain 0. The lowest scoring-tie id is LEFT.
    chosen = GreedyAgent().select_action(encode_board(env.board), env.legal_action_mask(), game=env.game)
    assert chosen == Action.LEFT
    np.testing.assert_array_equal(env.board, original.board)
    assert env.game._rng.getstate() == original._rng.getstate()
    assert (env.game.score, env.game.moves_made) == (original.score, original.moves_made)


@pytest.mark.parametrize("agent", [RandomAgent(0), GreedyAgent()])
def test_reference_policy_rejects_terminal_mask(agent):
    env = Env2048(seed=0)
    env.game.board = np.array([[2, 4, 2, 4], [4, 2, 4, 2]] * 2)
    with pytest.raises(ValueError, match="all-false"):
        agent.select_action(encode_board(env.board), env.legal_action_mask(), game=env.game)


@pytest.mark.parametrize("variant", ["random", "greedy"])
def test_reference_policies_use_existing_evaluator_reproducibly(variant):
    metrics = [evaluate(create_agent(variant, tiny_config()), episodes=2,
                        seed=1000, max_steps_per_episode=8) for _ in range(2)]
    assert metrics[0] == metrics[1]
    assert metrics[0]["mean_episode_length"] == 8


def test_summary_uses_sample_std_and_leaves_single_run_std_missing():
    rows = [{"agent": "a", "mean_score": 10.0}, {"agent": "a", "mean_score": 14.0},
            {"agent": "b", "mean_score": 100.0}]
    summary = summarize_results(rows, ["mean_score"])
    assert summary[0] == {"agent": "a", "runs": 2, "mean_score_mean": 12.0,
                          "mean_score_std": pytest.approx(np.sqrt(8))}
    assert summary[1]["mean_score_std"] is None


def test_full_comparison_has_separate_artifacts_and_is_reproducible(tmp_path):
    training = TrainingConfig(episodes=1, max_steps_per_episode=4,
                              epsilon_start=0.0, epsilon_end=0.0)
    settings = dict(variants=VARIANTS, seeds=(7, 8), training=training, dqn=tiny_config(),
                    evaluation_episodes=1, evaluation_seed=1000, evaluation_max_steps=4)
    first = run_comparison(tmp_path / "first", **settings)
    second = run_comparison(tmp_path / "second", **settings)
    assert first == second
    assert len(first) == 14
    manifest = json.loads((tmp_path / "first" / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["completed_runs"] == 14
    assert manifest["evaluation"]["seed"] == 1000
    assert manifest["seeds"] == [7, 8]
    with (tmp_path / "first" / "results.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 14
    with (tmp_path / "first" / "summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 7
    assert all(row["runs"] == "2" for row in summary)
    paths = []
    for row in first:
        with (tmp_path / "first" / row["evaluation_history"]).open() as stream:
            evaluation_rows = list(csv.DictReader(stream))
        assert len(evaluation_rows) == 1
        assert int(evaluation_rows[0]["seed"]) == 1000
        assert float(evaluation_rows[0]["score"]) == row["mean_score"]
        if row["agent"] in ("random", "greedy"):
            assert row["checkpoint"] is None and row["history"] is None
            assert row["learning_updates"] == row["training_steps"] == 0
            continue
        paths.append(row["checkpoint"])
        assert row["training_steps"] == 4 and row["learning_updates"] == 3
        path = tmp_path / "first" / row["checkpoint"]
        assert path.is_file()
        loaded = DQNAgent.load(path, device="cpu")
        assert loaded.config.seed == row["seed"]
        assert loaded.config.double_dqn == row["double_dqn"]
        assert loaded.config.dueling == row["dueling"]
        with (tmp_path / "first" / row["history"]).open() as stream:
            history = list(csv.DictReader(stream))
        assert len(history) == 1 and float(history[0]["steps"]) == 4
    assert len(paths) == len(set(paths)) == 10


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        run_comparison(output, variants=("random",), seeds=(7,), evaluation_episodes=1)
    assert marker.read_text() == "keep"
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize("settings", [
    {"variants": ()}, {"variants": ("dqn", "dqn")}, {"variants": ("unknown",)},
    {"seeds": ()}, {"seeds": (7, 7)}, {"seeds": (-1,)}, {"seeds": (1000,)},
    {"evaluation_max_steps": 0}, {"high_risk_threshold": float("nan")},
])
def test_invalid_comparisons_do_not_create_outputs(tmp_path, settings):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_comparison(output, **settings)
    assert not output.exists()


def test_warmup_loss_is_blank_and_partial_failure_is_recorded(tmp_path, monkeypatch):
    from training import compare

    output = tmp_path / "failure"
    original_evaluate = compare.evaluate

    def fail_second(agent, **kwargs):
        if isinstance(agent, GreedyAgent):
            raise RuntimeError("test evaluation failure")
        return original_evaluate(agent, **kwargs)

    monkeypatch.setattr(compare, "evaluate", fail_second)
    with pytest.raises(RuntimeError, match="test evaluation failure"):
        run_comparison(output, variants=("dqn", "greedy"), seeds=(7,),
                       training=TrainingConfig(episodes=1, max_steps_per_episode=1),
                       dqn=tiny_config(), evaluation_episodes=1, evaluation_max_steps=1)
    with (output / "dqn" / "seed_7" / "history.csv").open() as stream:
        assert list(csv.DictReader(stream))[0]["loss"] == ""
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["completed_runs"] == 1
    assert "test evaluation failure" in manifest["error"]
    assert (output / "results.csv").is_file()
