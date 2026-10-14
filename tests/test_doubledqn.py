"""Double DQN targets, baseline compatibility, and training integration."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl.agent import DQNAgent, DQNConfig
from rl.doubledqn import DoubleDQNAgent
from training.train import TrainingConfig, train


@pytest.mark.parametrize(
    "agent_class, double_dqn",
    [(DoubleDQNAgent, True), (DQNAgent, True), (DQNAgent, False)],
    ids=["double", "legacy-double", "standard"],
)
@pytest.mark.parametrize(
    "done, mask, double_next_q, standard_next_q",
    [
        (False, [False, True, True, True], 10.0, 30.0),
        (False, [False, False, False, True], 2.0, 2.0),
        (True, [False, True, True, True], 0.0, 0.0),
        (True, [False, False, False, False], 0.0, 0.0),
        (False, [False, False, False, False], 0.0, 0.0),
    ],
    ids=["different-best-actions", "one-legal-action", "terminal-with-legal-actions",
         "terminal-empty-mask", "nonterminal-empty-mask"],
)
def test_bellman_targets_and_gradient_isolation(
    monkeypatch, agent_class, double_dqn, done, mask, double_next_q, standard_next_q
):
    agent = agent_class(DQNConfig(
        hidden_dims=(16,), batch_size=2, min_replay_size=2, gamma=0.5,
        reward_scale=1.0, double_dqn=double_dqn, dueling=False,
        target_tau=None, target_update_interval=100, seed=7, device="cpu",
    ))
    # Both networks prefer illegal action 0; their best legal actions differ.
    with torch.no_grad():
        for network in (agent.online_net, agent.target_net):
            for parameter in network.parameters():
                parameter.zero_()
        agent.online_net.network[-1].bias.copy_(torch.tensor([100.0, 4.0, 8.0, 1.0]))
        agent.target_net.network[-1].bias.copy_(torch.tensor([200.0, 30.0, 10.0, 2.0]))

    state = np.zeros(16, dtype=np.float32)
    for reward in (2.0, 4.0):
        agent.observe(state, 0, reward, np.ones(16, dtype=np.float32), done, np.array(mask))
    online_before = [parameter.detach().clone() for parameter in agent.online_net.parameters()]
    target_before = [parameter.detach().clone() for parameter in agent.target_net.parameters()]
    captured_targets = []
    loss_forward = agent.loss_fn.forward

    def capture_loss(predicted, targets):
        assert predicted.requires_grad
        assert not targets.requires_grad
        assert targets.grad_fn is None
        captured_targets.append(targets.detach().clone())
        return loss_forward(predicted, targets)

    monkeypatch.setattr(agent.loss_fn, "forward", capture_loss)
    online_grad_modes = []
    target_grad_modes = []
    online_hook = agent.online_net.register_forward_hook(
        lambda *args: online_grad_modes.append(torch.is_grad_enabled())
    )
    target_hook = agent.target_net.register_forward_hook(
        lambda *args: target_grad_modes.append(torch.is_grad_enabled())
    )
    try:
        metrics = agent.learn()
    finally:
        online_hook.remove()
        target_hook.remove()

    next_q = double_next_q if double_dqn else standard_next_q
    assert len(captured_targets) == 1
    torch.testing.assert_close(
        captured_targets[0].sort().values, torch.tensor([2.0, 4.0]) + 0.5 * next_q
    )
    assert online_grad_modes == ([True, False] if double_dqn else [True])
    assert target_grad_modes == [False]
    assert all(parameter.grad is None for parameter in agent.target_net.parameters())
    for before, after in zip(target_before, agent.target_net.parameters()):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert any(not torch.equal(before, after) for before, after in zip(
        online_before, agent.online_net.parameters()
    ))
    assert metrics is not None
    assert all(np.isfinite(value) for value in metrics.values())
    assert agent.learn_steps == 1


def test_double_dqn_starts_fresh_without_mutating_config():
    config = DQNConfig(
        hidden_dims=(16,), batch_size=2, min_replay_size=2,
        double_dqn=False, seed=8, device="cpu",
    )
    agent = DoubleDQNAgent(config)
    assert config.double_dqn is False
    assert agent.config.double_dqn is True
    assert agent.learn_steps == 0
    assert len(agent.replay) == 0
    assert len(agent.optimizer.state) == 0
    for online, target in zip(agent.online_net.parameters(), agent.target_net.parameters()):
        torch.testing.assert_close(online, target, rtol=0, atol=0)


@pytest.mark.parametrize("dueling", [True, False])
def test_double_dqn_shared_training_and_checkpoint_roundtrip(tmp_path, dueling):
    agent = DoubleDQNAgent(DQNConfig(
        hidden_dims=(16,), batch_size=2, min_replay_size=2,
        replay_capacity=16, dueling=dueling, seed=8, device="cpu",
    ))
    checkpoint = tmp_path / "doubledqn.pt"
    history = train(
        TrainingConfig(episodes=1, max_steps_per_episode=8,
                       checkpoint_path=str(checkpoint), seed=8),
        agent.config, agent=agent,
    )
    assert len(history) == 1
    assert history[0]["steps"] == 8.0
    assert np.isfinite(history[0]["loss"])
    assert agent.learn_steps == 7
    for agent_class in (DoubleDQNAgent, DQNAgent):
        loaded = agent_class.load(checkpoint, device="cpu")
        assert loaded.config.double_dqn is True
        assert loaded.config.dueling is dueling
        assert loaded.learn_steps == agent.learn_steps
        assert len(loaded.optimizer.state) == len(agent.optimizer.state) > 0
        for name in ("online_net", "target_net"):
            for saved, restored in zip(
                getattr(agent, name).parameters(), getattr(loaded, name).parameters()
            ):
                torch.testing.assert_close(saved, restored, rtol=0, atol=0)


def test_shared_training_rejects_mismatched_agent_config(tmp_path):
    agent = DoubleDQNAgent(DQNConfig(hidden_dims=(16,), seed=8, device="cpu"))
    with pytest.raises(ValueError, match="must match"):
        train(TrainingConfig(checkpoint_path=str(tmp_path / "unused.pt")),
              DQNConfig(double_dqn=False), agent=agent)


def test_double_dqn_cli_trains_from_scratch(tmp_path, monkeypatch):
    from training import train_doubledqn

    checkpoint = tmp_path / "doubledqn.pt"

    def forbid_load(*args, **kwargs):
        pytest.fail("Fresh Double DQN training must not load a checkpoint.")

    monkeypatch.setattr(DoubleDQNAgent, "load", forbid_load)
    monkeypatch.setattr("sys.argv", [
        "train_doubledqn", "--episodes", "1", "--max-steps-per-episode", "4",
        "--batch-size", "2", "--min-replay-size", "2", "--device", "cpu",
        "--checkpoint", str(checkpoint),
    ])
    train_doubledqn.main()
    loaded = DQNAgent.load(checkpoint, device="cpu")
    assert loaded.config.double_dqn is True
    assert loaded.learn_steps == 3


