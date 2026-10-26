"""Compatibility of standalone Dueling checkpoints with shared entrypoints."""

import numpy as np
import pytest

pytest.importorskip("torch")

from game.environment import Env2048, encode_board
from inference import autoplay
from rl.agent import DQNAgent
from rl.duelingDQN import DuelingConfig, DuelingDQNAgent
from training.evaluate import evaluate


@pytest.mark.parametrize("max_exponent", [16.0, 8.0])
def test_standalone_dueling_checkpoint_shared_roundtrip(tmp_path, monkeypatch, max_exponent):
    original = DuelingDQNAgent(DuelingConfig(
        hidden_dims=(16,), batch_size=2, min_replay_size=2,
        replay_capacity=8, max_exponent=max_exponent, seed=7, device="cpu",
    ))
    env = Env2048(seed=7)
    state = env.reset()
    for _ in range(3):
        action = original.select_action(state, env.legal_action_mask())
        next_state, reward, done, _ = env.step(action)
        original.observe(state, action, reward, next_state, done, env.legal_action_mask())
        original.learn()
        state = next_state
    checkpoint = tmp_path / "dueling.pt"
    original.save(checkpoint)
    loaded = DQNAgent.load(checkpoint, device="cpu")
    assert loaded.config.dueling
    assert loaded.config.max_exponent == max_exponent
    assert loaded.learn_steps == original.learn_steps == 2
    np.testing.assert_allclose(loaded.q_values(state), original.q_values(state), rtol=1e-6)
    assert evaluate(loaded, episodes=1, max_steps_per_episode=4)["mean_episode_length"] == 4

    monkeypatch.setattr(autoplay, "_agent", None)
    monkeypatch.setenv("DQN_CHECKPOINT", str(checkpoint))
    action = autoplay.choose_action(env.game)
    assert action in env.game.legal_actions()
    np.testing.assert_allclose(autoplay._agent.q_values(encode_board(env.game.board)),
                               original.q_values(state), rtol=1e-6)

    shared_path = tmp_path / "shared.pt"
    loaded.save(shared_path)
    reloaded = DQNAgent.load(shared_path, device="cpu")
    assert reloaded.config.max_exponent == max_exponent
    np.testing.assert_allclose(reloaded.q_values(state), original.q_values(state), rtol=1e-6)
    assert len(reloaded.optimizer.state) == len(original.optimizer.state) > 0
