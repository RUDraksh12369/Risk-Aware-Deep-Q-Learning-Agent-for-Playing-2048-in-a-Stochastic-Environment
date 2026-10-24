"""Episode-boundary recovery for the Double DQN command."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from rl.agent import DQNConfig
from rl.doubledqn import DoubleDQNAgent
from training.train import TrainingConfig, train
from training import train_doubledqn


def test_resume_matches_uninterrupted_training(tmp_path):
    config = DQNConfig(hidden_dims=(16,), batch_size=2, min_replay_size=2,
                       replay_capacity=8, seed=7, device="cpu", dueling=False)
    training = TrainingConfig(episodes=4, max_steps_per_episode=8,
                              epsilon_decay_steps=25, seed=7, checkpoint_every=1,
                              checkpoint_path=str(tmp_path / "full.pt"))
    full = DoubleDQNAgent(config)
    expected = train(training, full.config, agent=full, resumable=True)
    split = DoubleDQNAgent(config)
    path = tmp_path / "split.pt"
    train(replace(training, episodes=2, checkpoint_path=str(path)), split.config,
          agent=split, resumable=True)
    extra = torch.load(path, weights_only=False)["extra"]
    resumed = DoubleDQNAgent.load(path, device="cpu")
    actual = train(replace(training, checkpoint_path=str(path)), resumed.config,
                   agent=resumed, resume=extra, resumable=True)
    assert actual == expected
    assert resumed.learn_steps == full.learn_steps == 31
    assert len(resumed.replay) == 8
    for name in ("online_net", "target_net"):
        for left, right in zip(getattr(full, name).parameters(), getattr(resumed, name).parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_cli_auto_resume_and_completed_run_noop(tmp_path, monkeypatch):
    path = tmp_path / "auto.pt"
    arguments = ["train_doubledqn", "--checkpoint", str(path), "--device", "cpu",
                 "--max-steps-per-episode", "4", "--batch-size", "2",
                 "--min-replay-size", "2", "--no-dueling"]
    monkeypatch.setattr("sys.argv", arguments + ["--episodes", "1"])
    train_doubledqn.main()
    monkeypatch.setattr("sys.argv", arguments + ["--episodes", "2"])
    train_doubledqn.main()
    payload = torch.load(path, weights_only=False)
    assert payload["extra"]["episode"] == 2
    assert payload["extra"]["global_step"] == 8
    assert payload["learn_steps"] == 7
    before = path.read_bytes()
    train_doubledqn.main()
    assert path.read_bytes() == before
    monkeypatch.setattr("sys.argv", arguments + ["--episodes", "1", "--fresh"])
    train_doubledqn.main()
    assert torch.load(path, weights_only=False)["learn_steps"] == 3


def test_failed_atomic_save_preserves_checkpoint(tmp_path, monkeypatch):
    config = DQNConfig(hidden_dims=(16,), batch_size=2, min_replay_size=2, device="cpu")
    agent = DoubleDQNAgent(config)
    path = tmp_path / "existing.pt"
    agent.save(path)
    original = path.read_bytes()

    def fail_save(destination, extra):
        destination.write_bytes(b"incomplete")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(agent, "save", fail_save)
    with pytest.raises(OSError, match="interrupted"):
        train(TrainingConfig(episodes=1, checkpoint_path=str(path)), agent.config,
              agent=agent, resumable=True)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_legacy_resume_and_missing_source(tmp_path, monkeypatch, capsys):
    path = tmp_path / "legacy.pt"
    agent = DoubleDQNAgent(DQNConfig(hidden_dims=(16,), batch_size=2,
                                    min_replay_size=2, device="cpu"))
    train(TrainingConfig(episodes=1, max_steps_per_episode=4, checkpoint_path=str(path)),
          agent.config, agent=agent)
    monkeypatch.setattr("sys.argv", ["train_doubledqn", "--resume", str(path),
                        "--checkpoint", str(tmp_path / "new.pt"), "--episodes", "2",
                        "--max-steps-per-episode", "4", "--device", "cpu"])
    train_doubledqn.main()
    assert "legacy checkpoint" in capsys.readouterr().out
    saved = torch.load(tmp_path / "new.pt", weights_only=False)
    assert saved["extra"]["episode"] == 2
    assert saved["extra"]["global_step"] == 8
    monkeypatch.setattr("sys.argv", ["train_doubledqn", "--resume", str(tmp_path / "missing.pt")])
    with pytest.raises(SystemExit):
        train_doubledqn.main()


def test_interrupted_episode_recovers_last_completed_boundary(tmp_path, monkeypatch):
    agent = DoubleDQNAgent(DQNConfig(hidden_dims=(16,), batch_size=2,
                                    min_replay_size=2, seed=7, device="cpu"))
    path = tmp_path / "interrupted.pt"
    training = TrainingConfig(episodes=2, max_steps_per_episode=4,
                              checkpoint_every=1, checkpoint_path=str(path), seed=7)
    learn = agent.learn
    calls = 0

    def interrupted_learn():
        nonlocal calls
        calls += 1
        if calls == 6:
            raise KeyboardInterrupt
        return learn()

    monkeypatch.setattr(agent, "learn", interrupted_learn)
    with pytest.raises(KeyboardInterrupt):
        train(training, agent.config, agent=agent, resumable=True)
    extra = torch.load(path, weights_only=False)["extra"]
    assert extra["episode"] == 1
    assert extra["global_step"] == 4
    resumed = DoubleDQNAgent.load(path, device="cpu")
    history = train(training, resumed.config, agent=resumed, resume=extra, resumable=True)
    assert [record["episode"] for record in history] == [1, 2]
    assert resumed.learn_steps == 7
    assert len(resumed.replay) == 8
