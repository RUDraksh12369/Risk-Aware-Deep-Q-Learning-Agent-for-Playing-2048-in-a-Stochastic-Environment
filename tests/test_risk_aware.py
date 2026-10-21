"""RA-D3QN policy integration, baseline equivalence, and checkpoint safety."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from game.environment import Env2048, encode_board
from game.game import Action
from rl.agent import DQNAgent, DQNConfig
from rl.doubledqn import DoubleDQNAgent
from rl.risk import RiskConfig
from rl.risk_aware import RiskAwareConfig, RiskAwareDQNAgent
from training.evaluate import evaluate
from training.train import TrainingConfig, train


def small_config():
    return DQNConfig(hidden_dims=(16,), batch_size=2, min_replay_size=2,
                     replay_capacity=32, seed=7, device="cpu")


def decision_env():
    env = Env2048(seed=7)
    env.game.board = np.array([
        [2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [2, 2, 4, 0],
    ], dtype=np.int64)
    return env


def known_policy():
    agent = RiskAwareDQNAgent(small_config(), RiskAwareConfig(1.0))
    with torch.no_grad():
        for parameter in agent.online_net.parameters():
            parameter.zero_()
        agent.online_net.advantage_stream.bias.copy_(torch.tensor([100.0, 1.01, 1.0, 0.5]))
    return agent


def test_risk_changes_exploitation_but_preserves_live_game():
    env = decision_env()
    state = encode_board(env.board)
    agent = known_policy()
    before = env.game.clone()
    assert agent.select_action(state, env.legal_action_mask(), game=env.game) == Action.LEFT
    baseline = DoubleDQNAgent(small_config())
    baseline.online_net.load_state_dict(agent.online_net.state_dict())
    assert baseline.select_action(state, env.legal_action_mask()) == Action.DOWN
    np.testing.assert_array_equal(env.board, before.board)
    assert env.game._rng.getstate() == before._rng.getstate()
    assert (env.game.score, env.game.moves_made, env.done) == (before.score, before.moves_made, before.done)


def test_exploration_uses_legal_actions_without_requiring_risk_values(monkeypatch):
    agent = known_policy()
    env = decision_env()

    def forbid_values(*args):
        pytest.fail("Exploration should not evaluate the networks or risks.")

    monkeypatch.setattr(agent, "_policy_q_values", forbid_values)
    actions = {agent.select_action(encode_board(env.board), env.legal_action_mask(), 1.0)
               for _ in range(50)}
    assert actions == {Action.DOWN, Action.LEFT, Action.RIGHT}


def test_risk_exploitation_requires_matching_board_and_mask():
    agent = known_policy()
    env = decision_env()
    state = encode_board(env.board)
    mask = env.legal_action_mask()
    with pytest.raises(ValueError, match="requires game"):
        agent.select_action(state, mask)
    with pytest.raises(ValueError, match="state must encode"):
        agent.select_action(np.zeros(16), mask, game=env.game)
    with pytest.raises(ValueError, match="legal_mask must match"):
        agent.select_action(state, np.ones(4, dtype=bool), game=env.game)


def test_zero_lambda_matches_baseline_actions_including_exploration():
    baseline = DoubleDQNAgent(small_config())
    agent = RiskAwareDQNAgent(small_config(), RiskAwareConfig(0.0))
    env = decision_env()
    state = encode_board(env.board)
    for epsilon in (0.0, 0.5, 1.0):
        for _ in range(10):
            assert agent.select_action(state, env.legal_action_mask(), epsilon, game=env.game) == (
                baseline.select_action(state, env.legal_action_mask(), epsilon)
            )


@pytest.mark.parametrize("coefficient", [0.0, 10.0])
def test_policy_risk_does_not_change_rewards_or_double_dqn_learning_targets(coefficient):
    baseline = DoubleDQNAgent(small_config())
    agent = RiskAwareDQNAgent(small_config(), RiskAwareConfig(coefficient))
    state = np.zeros(16, dtype=np.float32)
    next_state = np.ones(16, dtype=np.float32)
    for model in (baseline, agent):
        model.observe(state, 0, 100.0, next_state, False, np.array([True, False, True, False]))
        model.observe(next_state, 2, 200.0, state, True, np.zeros(4, dtype=bool))
        np.testing.assert_array_equal(np.sort(model.replay.sample(2).rewards), [1.0, 2.0])
    assert agent.learn() == baseline.learn()
    for name in ("online_net", "target_net"):
        for actual, expected in zip(getattr(agent, name).parameters(), getattr(baseline, name).parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in agent.target_net.parameters())


def test_constructor_enables_dueling_double_without_mutating_callers_config():
    config = small_config()
    config.dueling = False
    config.double_dqn = False
    agent = RiskAwareDQNAgent(config)
    assert agent.config.dueling and agent.config.double_dqn
    assert not config.dueling and not config.double_dqn


@pytest.mark.parametrize("coefficient", [-1, float("nan"), float("inf")])
def test_invalid_penalties_are_rejected(coefficient):
    with pytest.raises(ValueError, match="risk_coefficient"):
        RiskAwareConfig(coefficient)


def test_training_checkpoint_restores_risk_policy_and_optimizer(tmp_path):
    agent = RiskAwareDQNAgent(small_config(), RiskAwareConfig(0.7, RiskConfig(2, 3, 4)))
    path = tmp_path / "risk.pt"
    history = train(TrainingConfig(
        episodes=1, max_steps_per_episode=8, epsilon_start=0.0, epsilon_end=0.0,
        checkpoint_path=str(path), seed=7,
    ), agent.config, agent=agent)
    assert len(history) == 1 and history[0]["steps"] == 8
    assert np.isfinite(history[0]["loss"])
    assert agent.learn_steps == 7
    env = decision_env()
    state = encode_board(env.board)
    for loader in (DQNAgent, RiskAwareDQNAgent):
        restored = loader.load(path, device="cpu")
        assert isinstance(restored, RiskAwareDQNAgent)
        assert restored.risk_config == agent.risk_config
        assert restored.learn_steps == agent.learn_steps
        assert len(restored.optimizer.state) == len(agent.optimizer.state) > 0
        np.testing.assert_array_equal(restored.q_values(state), agent.q_values(state))
        assert restored.select_action(state, env.legal_action_mask(), game=env.game) == (
            agent.select_action(state, env.legal_action_mask(), game=env.game)
        )
    with pytest.raises(ValueError, match="RA-D3QN checkpoint"):
        DoubleDQNAgent.load(path, device="cpu")


def test_baseline_checkpoint_does_not_silently_acquire_risk(tmp_path):
    path = tmp_path / "baseline.pt"
    baseline = DQNAgent(small_config())
    baseline.save(path)
    assert type(DQNAgent.load(path, device="cpu")) is DQNAgent
    with pytest.raises(ValueError, match="saved risk settings"):
        RiskAwareDQNAgent.load(path, device="cpu")


def test_gui_inference_restores_and_uses_saved_risk_policy(tmp_path, monkeypatch):
    from inference import autoplay

    path = tmp_path / "risk.pt"
    known_policy().save(path)
    monkeypatch.setattr(autoplay, "_agent", None)
    monkeypatch.setenv("DQN_CHECKPOINT", str(path))
    assert autoplay.choose_action(decision_env().game) == Action.LEFT
    assert isinstance(autoplay._agent, RiskAwareDQNAgent)


def test_evaluation_is_repeatable_and_counts_truncation_separately():
    agent = RiskAwareDQNAgent(small_config())
    metrics = evaluate(agent, episodes=2, seed=9, max_steps_per_episode=1, high_risk_threshold=0.0)
    assert metrics == evaluate(agent, episodes=2, seed=9, max_steps_per_episode=1, high_risk_threshold=0.0)
    assert metrics["mean_episode_length"] == 1
    assert metrics["game_over_rate"] == 0
    assert metrics["truncation_rate"] == 1
    assert metrics["tile_2048_rate"] == 0
    assert metrics["high_risk_rate"] == 1
    assert metrics["mean_empty_cells"] == 14
    assert 1 <= metrics["mean_legal_moves"] <= 4
    assert all(np.isfinite(value) for value in metrics.values())


def test_evaluation_counts_terminal_outcome_and_pre_action_risk(monkeypatch):
    from training import evaluate as evaluation_module

    def make_scripted_env(seed):
        env = Env2048(seed=seed)

        def reset():
            env.game.board = np.zeros((4, 4), dtype=np.int64)
            env.game.board[1, 1] = 2
            return encode_board(env.board)

        def step(action):
            env.game.board = np.array([[1024, 2048, 1024, 2048], [2048, 1024, 2048, 1024]] * 2)
            env.game.done = True
            env.game.score = 100
            return encode_board(env.board), 100.0, True, {}

        env.reset = reset
        env.step = step
        return env

    monkeypatch.setattr(evaluation_module, "Env2048", make_scripted_env)
    metrics = evaluate(RiskAwareDQNAgent(small_config()), episodes=1, max_steps_per_episode=1)
    assert metrics["mean_score"] == 100
    assert metrics["tile_2048_rate"] == 1
    assert metrics["max_tile"] == 2048
    assert metrics["game_over_rate"] == 1
    assert metrics["truncation_rate"] == 0
    assert metrics["mean_legal_moves"] == 4
    assert metrics["mean_empty_cells"] == 15
    assert metrics["mean_board_risk"] == pytest.approx(1 / 48)
    assert metrics["high_risk_rate"] == 0
    assert metrics["low_mobility_rate"] == 0


def test_risk_training_cli_creates_fresh_checkpoint(tmp_path, monkeypatch):
    from training import train_risk_aware

    path = tmp_path / "risk.pt"
    monkeypatch.setattr("sys.argv", [
        "train_risk_aware", "--episodes", "1", "--max-steps-per-episode", "4",
        "--batch-size", "2", "--min-replay-size", "2", "--device", "cpu",
        "--risk-coefficient", "0.3", "--mobility-weight", "2",
        "--epsilon-start", "0", "--epsilon-end", "0", "--checkpoint", str(path),
    ])
    train_risk_aware.main()
    agent = DQNAgent.load(path, device="cpu")
    assert isinstance(agent, RiskAwareDQNAgent)
    assert agent.learn_steps == 3
    assert agent.risk_config == RiskAwareConfig(0.3, RiskConfig(2, 1, 1))
