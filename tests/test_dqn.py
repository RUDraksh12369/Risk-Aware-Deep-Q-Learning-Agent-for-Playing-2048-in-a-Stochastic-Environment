"""Focused tests for the DQN components (skipped when optional torch is absent)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl.agent import DQNAgent, DQNConfig
from rl.dqn import DQN
from rl.replay_buffer import ReplayBuffer
from training.evaluate import evaluate


def test_network_outputs_one_value_per_action():
    network = DQN(state_dim=16, n_actions=4, hidden_dims=(32,))
    output = network(torch.zeros((3, 16)))
    assert output.shape == (3, 4)


def test_original_mlp_mode_remains_available_for_old_checkpoints():
    network = DQN(state_dim=16, n_actions=4, hidden_dims=(32,), dueling=False)
    assert network(torch.zeros(16)).shape == (1, 4)


def test_action_selection_never_returns_a_masked_action():
    agent = DQNAgent(DQNConfig(hidden_dims=(16,), batch_size=2, min_replay_size=2, seed=3, device="cpu"))
    state = np.zeros(16, dtype=np.float32)
    mask = np.array([False, True, False, True])
    for _ in range(10):
        assert agent.select_action(state, mask, epsilon=1.0) in (1, 3)


def test_replay_and_learning_update():
    agent = DQNAgent(DQNConfig(hidden_dims=(16,), batch_size=2, min_replay_size=2, seed=4, device="cpu"))
    state = np.zeros(16, dtype=np.float32)
    next_state = np.ones(16, dtype=np.float32)
    agent.observe(state, 0, 4.0, next_state, False, np.array([True, False, True, False]))
    agent.observe(next_state, 2, 8.0, state, True, np.zeros(4, dtype=bool))
    metrics = agent.learn()
    assert metrics is not None
    assert metrics["loss"] >= 0.0
    assert agent.learn_steps == 1


def test_learning_is_finite_when_a_nonterminal_transition_has_no_legal_move():
    agent = DQNAgent(DQNConfig(hidden_dims=(16,), batch_size=2, min_replay_size=2, seed=5, device="cpu"))
    state = np.zeros(16, dtype=np.float32)
    agent.observe(state, 0, 4.0, state, False, np.zeros(4, dtype=bool))
    agent.observe(state, 1, 8.0, state, True, np.zeros(4, dtype=bool))
    metrics = agent.learn()
    assert metrics is not None
    assert np.isfinite(metrics["loss"])


def test_reward_scaling_is_applied_only_to_the_learning_signal():
    agent = DQNAgent(DQNConfig(hidden_dims=(16,), batch_size=2, min_replay_size=2, reward_scale=0.01, device="cpu"))
    state = np.zeros(16, dtype=np.float32)
    agent.observe(state, 0, 100.0, state, True, np.zeros(4, dtype=bool))
    assert agent.replay.sample(1).rewards[0] == pytest.approx(1.0)


def test_greedy_evaluation_returns_aggregate_metrics():
    agent = DQNAgent(DQNConfig(hidden_dims=(16,), batch_size=2, min_replay_size=2, seed=6, device="cpu"))
    metrics = evaluate(agent, episodes=2, seed=99)
    assert metrics["episodes"] == 2.0
    assert metrics["mean_score"] >= 0.0
    assert metrics["mean_max_tile"] >= 2.0


def test_replay_buffer_has_fixed_capacity():
    buffer = ReplayBuffer(capacity=2, seed=0)
    for index in range(3):
        buffer.add(np.full(2, index), 0, 0.0, np.zeros(2), False, np.ones(4, dtype=bool))
    assert len(buffer) == 2
    assert buffer.sample(2).states.shape == (2, 2)


