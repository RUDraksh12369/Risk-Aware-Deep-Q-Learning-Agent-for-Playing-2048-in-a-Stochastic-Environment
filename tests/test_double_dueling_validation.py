"""DDDQN checks that never train, play games, or write checkpoints."""

import io
import random
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from rl import double_dueling_dqn as module
from rl.agent import DQNAgent, DQNConfig
from rl.replay_buffer import TransitionBatch
from training.train import TrainingConfig
from tests import test_double_dqn_validation as shared


class DoubleDuelingValidation(unittest.TestCase):
    setUp = shared.NonTrainingValidation.setUp

    def invoke(self, args):
        with patch("sys.argv", ["double_dueling_dqn", *args]):
            module.main()

    def payload(self, agent):
        return {"format": "double_dueling_dqn_v1", "config": asdict(agent.config), "extra": {
            "episode": 3, "global_step": 30, "history": [],
            "training_config": asdict(TrainingConfig(episodes=3, max_steps_per_episode=10,
                                                     epsilon_start=.4, epsilon_end=.2)),
            "training_state": {"replay": agent.replay, "policy_rng": agent._rng.getstate(),
                               "environment_rng": random.Random(7).getstate(),
                               "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                               "torch_rng": torch.get_rng_state(), "cuda_rng": None},
        }}

    def test_dueling_heads_double_targets_and_forced_flags(self):
        config = replace(self.config, double_dqn=False, dueling=False, gamma=.5)
        agent = module.DoubleDuelingDQNAgent(config)
        self.assertFalse(config.double_dqn)
        self.assertFalse(config.dueling)
        self.assertTrue(agent.config.double_dqn and agent.config.dueling)
        with torch.no_grad():
            for net, values in ((agent.online_net, [100., 4., 8., 1.]),
                                (agent.target_net, [200., 30., 10., 2.])):
                for p in net.parameters():
                    p.zero_()
                q = torch.tensor(values)
                net.value_stream.bias.fill_(q.mean())
                net.advantage_stream.bias.copy_(q)
        states = np.zeros((3, 16), np.float32)
        self.assertEqual(agent.online_net(torch.from_numpy(states)).shape, (3, 4))
        batch = TransitionBatch(states, np.ones(3, np.int64), np.full(3, 2., np.float32), states,
                                np.array([0, 1, 0]), np.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]], bool))
        loss, predicted, targets = agent.compute_td_loss(batch)
        torch.testing.assert_close(targets, torch.tensor([7., 2., 2.]))
        torch.testing.assert_close(predicted, torch.full((3,), 4.))
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(targets.requires_grad)
        self.assertEqual(agent.learn_steps, 0)
        self.assertFalse(agent.optimizer.state)
        self.assertTrue(all(p.grad is None for p in agent.target_net.parameters()))

    def test_fresh_cli_configuration_propagation(self):
        with patch.object(Path, "exists", return_value=False), patch.object(module, "train") as train:
            self.invoke(["train", "--device", "cpu", "--hidden-dims", "12", "6",
                         "--max-exponent", "8", "--reward-scale", ".2", "--gradient-clip-norm", "3",
                         "--hard-target-updates", "--target-update-interval", "7"])
        training, config = train.call_args.args
        self.assertEqual(config.hidden_dims, (12, 6))
        self.assertEqual(config.max_exponent, 8.)
        self.assertEqual(config.reward_scale, .2)
        self.assertEqual(config.gradient_clip_norm, 3.)
        self.assertIsNone(config.target_tau)
        self.assertEqual(config.target_update_interval, 7)
        self.assertTrue(config.dueling and config.double_dqn)
        self.assertEqual(training.epsilon_start, 1.)

    def test_invalid_fresh_configuration_never_constructs_agent(self):
        cases = [["--seed", "-1"], ["--reward-scale", "nan"], ["--hidden-dims", "0"],
                 ["--target-update-interval", "0"], ["--max-exponent", "inf"],
                 ["--gradient-clip-norm", "nan"], ["--epsilon-decay-steps", "0"],
                 ["--epsilon-start", ".1", "--epsilon-end", ".9"]]
        for args in cases:
            with self.subTest(args=args), patch.object(Path, "exists", return_value=False), \
                    patch.object(module, "DoubleDuelingDQNAgent") as constructor, \
                    patch.object(module, "train") as train, self.assertRaises(SystemExit):
                self.invoke(["train", *args])
            constructor.assert_not_called()
            train.assert_not_called()

    def test_resume_saved_settings_and_noop(self):
        agent = module.DoubleDuelingDQNAgent(replace(self.config, reward_scale=.3, target_tau=None))
        payload = self.payload(agent)
        with patch.object(Path, "exists", return_value=False), patch.object(Path, "is_file", return_value=True), \
                patch("torch.load", return_value=payload), patch.object(module.DoubleDuelingDQNAgent, "load", return_value=agent), \
                patch.object(module, "train") as train:
            self.invoke(["train", "--resume", "source.pt", "--checkpoint", "new.pt", "--episodes", "5"])
        training, config = train.call_args.args
        self.assertEqual(training.episodes, 5)
        self.assertEqual(training.max_steps_per_episode, 10)
        self.assertEqual(training.epsilon_start, .4)
        self.assertEqual(config.reward_scale, .3)
        self.assertIsNone(config.target_tau)
        with patch.object(Path, "exists", return_value=False), patch.object(Path, "is_file", return_value=True), \
                patch("torch.load", return_value=payload), patch.object(module.DoubleDuelingDQNAgent, "load") as load, \
                patch.object(module, "train") as train:
            self.invoke(["train", "--resume", "source.pt", "--episodes", "3"])
            load.assert_not_called()
            train.assert_not_called()

    def test_malformed_resume_and_existing_outputs_rejected(self):
        agent = module.DoubleDuelingDQNAgent(self.config)
        cases = [[], {"format": "wrong"}, self.payload(agent), self.payload(agent), self.payload(agent)]
        cases[2]["extra"]["training_state"] = {}
        cases[3]["extra"]["episode"] = -1
        cases[4]["extra"]["training_config"]["epsilon_decay_steps"] = 0
        for payload in cases:
            with self.subTest(payload_type=type(payload)), patch.object(Path, "exists", return_value=False), \
                    patch.object(Path, "is_file", return_value=True), patch("torch.load", return_value=payload), \
                    patch.object(module.DoubleDuelingDQNAgent, "load") as load, \
                    patch.object(module, "train") as train, self.assertRaises(SystemExit):
                self.invoke(["train", "--resume", "source.pt"])
            load.assert_not_called()
            train.assert_not_called()
        with patch.object(Path, "exists", return_value=True), patch.object(module, "train") as train, \
                self.assertRaises(SystemExit):
            self.invoke(["train"])
        train.assert_not_called()

    def test_evaluation_validates_before_loading_and_delegates(self):
        with patch.object(module.DoubleDuelingDQNAgent, "load") as load, self.assertRaises(SystemExit):
            self.invoke(["evaluate", "--episodes", "0"])
        load.assert_not_called()
        with patch.object(module.DoubleDuelingDQNAgent, "load") as load, \
                patch.object(module, "evaluate", return_value={"mean_score": 0.}) as evaluate:
            self.invoke(["evaluate", "--episodes", "2", "--seed", "1001", "--max-steps-per-episode", "9"])
        evaluate.assert_called_once_with(load.return_value, 2, 1001, max_steps_per_episode=9)

    def test_in_memory_checkpoint_shared_loader_compatibility(self):
        agent = module.DoubleDuelingDQNAgent(self.config)
        buffer = io.BytesIO()
        save = torch.save
        with patch("torch.save", side_effect=lambda payload, _: save(payload, buffer)), patch.object(Path, "mkdir"):
            agent.save("not_written.pt")
        buffer.seek(0)
        payload = torch.load(buffer, weights_only=False)
        self.assertEqual(payload["format"], "double_dueling_dqn_v1")
        for loader in (DQNAgent, module.DoubleDuelingDQNAgent):
            with patch("torch.load", return_value=payload):
                restored = loader.load("not_written.pt", "cpu")
            np.testing.assert_allclose(agent.q_values(np.zeros(16)), restored.q_values(np.zeros(16)))
        with self.assertRaises(ValueError):
            module.DoubleDuelingDQNAgent._from_checkpoint(replace(agent.config, double_dqn=False), payload)

    def test_seed_validation_shared_model_and_environment_config(self):
        for seed in (-1, 2**32, True, 1.5):
            for config in (replace(self.config, seed=seed), TrainingConfig(seed=seed)):
                with self.subTest(seed=seed), self.assertRaises(ValueError):
                    config.validate()


if __name__ == "__main__":
    unittest.main()
