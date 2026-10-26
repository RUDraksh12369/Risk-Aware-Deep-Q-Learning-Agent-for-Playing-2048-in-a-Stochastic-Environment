"""Non-training regression checks: no environment steps or optimizer updates.

Run independently with: py -B -m unittest tests.test_double_dqn_validation -v
"""

import io
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from game.environment import Env2048, encode_board
from inference import autoplay
from rl.agent import DQNAgent, DQNConfig
from rl.dqn import DQN
from rl.doubledqn import DoubleDQNAgent, double_dqn_next_q
from rl.replay_buffer import ReplayBuffer, TransitionBatch
from training.train import TrainingConfig, epsilon_by_step, train
from training import train_doubledqn


class NonTrainingValidation(unittest.TestCase):
    def setUp(self):
        # Fail immediately if a future edit accidentally turns these into training tests.
        for target in ("game.environment.Env2048.step", "rl.agent.DQNAgent.learn",
                       "torch.optim.Adam.step", "torch.Tensor.backward"):
            blocker = patch(target, side_effect=AssertionError("Training is forbidden in this suite"))
            blocker.start()
            self.addCleanup(blocker.stop)
        self.config = DQNConfig(hidden_dims=(8,), batch_size=2, min_replay_size=2,
                                replay_capacity=8, seed=7, device="cpu", dueling=False)

    def test_config_rejects_invalid_values_before_construction(self):
        cases = {"batch_size": [0, -1, 1.5, True], "min_replay_size": [0, 1],
                 "target_update_interval": [0, 2.5], "learning_rate": [0, float("nan"), float("inf")],
                 "gradient_clip_norm": [0, float("nan"), float("inf")],
                 "reward_scale": [0, float("nan"), float("inf")],
                 "max_exponent": [0, float("nan"), float("inf")],
                 "gamma": [-1, 2, float("nan")], "target_tau": [0, 2, float("nan")],
                 "hidden_dims": [(), (0,), (2.5,)]}
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    DoubleDQNAgent(replace(self.config, **{key: value}))

    def test_encoding_normalization_and_dueling_forward(self):
        board = np.array([[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 0, 0]])
        state = encode_board(board)
        self.assertEqual(state.shape, (16,))
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_array_equal(state[:4], [0, 1, 2, 3])
        for dueling in (False, True):
            net = DQN(hidden_dims=(8,), max_exponent=8., dueling=dueling)
            x = torch.from_numpy(np.stack([state, state]))
            layer = net.feature_extractor if dueling else net.network
            inputs = []
            hook = layer[0].register_forward_pre_hook(lambda _, args: inputs.append(args[0]))
            output = net(x)
            hook.remove()
            torch.testing.assert_close(inputs[0], x / 8.)
            self.assertEqual(output.shape, (2, 4))
            if dueling:
                f = net.feature_extractor(x / 8.)
                adv = net.advantage_stream(f)
                torch.testing.assert_close(output, net.value_stream(f) + adv - adv.mean(1, keepdim=True))
        for scale in (0., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                DQN(max_exponent=scale)

    def test_exact_double_targets_terminal_truncation_and_detachment(self):
        agent = DoubleDQNAgent(replace(self.config, gamma=.5, target_tau=None))
        with torch.no_grad():
            for net, values in ((agent.online_net, [100., 4., 8., 1.]),
                                (agent.target_net, [200., 30., 10., 2.])):
                for p in net.parameters():
                    p.zero_()
                net.network[-1].bias.copy_(torch.tensor(values))
        masks = np.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=bool)
        batch = TransitionBatch(np.zeros((4, 16), np.float32), np.ones(4, np.int64),
                                np.full(4, 2., np.float32), np.ones((4, 16), np.float32),
                                np.array([0, 1, 0, 0], np.float32), masks)
        before = {key: value.clone() for key, value in agent.online_net.state_dict().items()}
        loss, predicted, targets = agent.compute_td_loss(batch)
        torch.testing.assert_close(targets, torch.tensor([7., 2., 2., 3.]))
        self.assertFalse(targets.requires_grad)
        self.assertIsNone(targets.grad_fn)
        self.assertTrue(predicted.requires_grad)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(agent.learn_steps, 0)
        self.assertEqual(len(agent.optimizer.state), 0)
        self.assertTrue(all(p.grad is None for p in agent.target_net.parameters()))
        for key, value in agent.online_net.state_dict().items():
            torch.testing.assert_close(value, before[key])
        raw = double_dqn_next_q(agent.online_net, agent.target_net,
                                torch.from_numpy(batch.next_states), torch.from_numpy(masks))
        self.assertEqual(raw[0].item(), 10.)  # Target max would incorrectly give 30.
        standard = DQNAgent(replace(agent.config, double_dqn=False))
        standard.online_net.load_state_dict(agent.online_net.state_dict())
        standard.target_net.load_state_dict(agent.target_net.state_dict())
        torch.testing.assert_close(standard.compute_td_loss(batch)[2], torch.tensor([17., 2., 2., 3.]))
        with self.assertRaises(ValueError):
            agent.compute_td_loss(replace(batch, rewards=batch.rewards[:, None]))
        with self.assertRaises(ValueError):
            agent.compute_td_loss(replace(batch, actions=batch.actions.astype(float) + .5))

    def test_replay_validation_copying_and_scaling(self):
        agent = DoubleDQNAgent(self.config)
        state = np.zeros(16, np.float32)
        mask = np.ones(4, bool)
        agent.observe(state, 1, 100., state, False, mask)
        state[:] = 9
        mask[:] = False
        batch = agent.replay.sample(1)
        np.testing.assert_array_equal(batch.states, np.zeros((1, 16)))
        self.assertAlmostEqual(batch.rewards[0], 1.)
        self.assertEqual(batch.actions.dtype, np.int64)
        self.assertEqual(batch.next_legal_masks.dtype, np.bool_)
        for changes in ({"action": 1.5}, {"action": -1}, {"action": 4},
                        {"reward": float("nan")}, {"state": np.zeros((1, 16))},
                        {"next_state": np.zeros(4)}, {"next_legal_mask": [1, 0, 2, 0]}):
            args = dict(state=np.zeros(16), action=0, reward=1., next_state=np.zeros(16),
                        done=False, next_legal_mask=np.ones(4, bool))
            args.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ReplayBuffer(8).add(**args)
        with self.assertRaises(ValueError):
            agent.observe(np.zeros(16), 0, 1., np.zeros(16), False, np.ones(3))

    def test_invalid_states_rejected_even_during_random_exploration(self):
        agent = DoubleDQNAgent(self.config)
        for epsilon in (0., 1.):
            for state in (np.zeros((1, 16)), np.full(16, np.nan)):
                with self.assertRaises(ValueError):
                    agent.select_action(state, np.ones(4, bool), epsilon)

    def test_target_soft_and_hard_sync_without_optimizer_steps(self):
        for tau in (.25, None):
            agent = DoubleDQNAgent(replace(self.config, target_tau=tau, target_update_interval=2))
            with torch.no_grad():
                for p in agent.online_net.parameters():
                    p.fill_(4.)
                for p in agent.target_net.parameters():
                    p.zero_()
            agent.learn_steps = 1
            agent.update_target_network()
            for p in agent.target_net.parameters():
                torch.testing.assert_close(p, torch.full_like(p, 1. if tau else 0.))
            if tau is None:
                agent.learn_steps = 2
                agent.update_target_network()
                for p in agent.target_net.parameters():
                    torch.testing.assert_close(p, torch.full_like(p, 4.))

    def test_training_config_validation_and_schedule_no_training(self):
        for changes in ({"epsilon_start": -1}, {"epsilon_end": float("nan")},
                        {"epsilon_decay_steps": 0}, {"checkpoint_every": 0},
                        {"episodes": 1.5}, {"game_over_penalty": float("inf")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(TrainingConfig(), **changes).validate()
        config = TrainingConfig(epsilon_start=1., epsilon_end=.1, epsilon_decay_steps=10)
        self.assertAlmostEqual(epsilon_by_step(5, config), .55)
        self.assertAlmostEqual(epsilon_by_step(20, config), .1)
        with self.assertRaises(ValueError):
            train(replace(config, epsilon_decay_steps=0), self.config)
        # The mismatch fails before agent creation, stepping or checkpoint I/O.
        with self.assertRaisesRegex(ValueError, "dimensions"):
            train(config, replace(self.config, state_dim=8))

    def test_cli_propagates_config_without_invoking_training(self):
        args = ["train_doubledqn", "--checkpoint", "unused_validation_output.pt", "--device", "cpu",
                "--hard-target-updates", "--target-update-interval", "7", "--reward-scale", ".1",
                "--gradient-clip-norm", "2", "--max-exponent", "8", "--hidden-dims", "12", "6"]
        with patch("sys.argv", args), patch.object(Path, "exists", return_value=False), \
                patch.object(train_doubledqn, "train") as blocked_train:
            train_doubledqn.main()
        _, model = blocked_train.call_args.args
        self.assertIsNone(model.target_tau)
        self.assertEqual(model.target_update_interval, 7)
        self.assertEqual(model.hidden_dims, (12, 6))
        self.assertEqual(model.reward_scale, .1)
        self.assertEqual(model.gradient_clip_norm, 2.)
        self.assertEqual(model.max_exponent, 8.)

    def test_checkpoint_roundtrip_in_memory_only(self):
        agent = DoubleDQNAgent(replace(self.config, max_exponent=8., target_tau=None))
        buffer = io.BytesIO()
        real_save = torch.save
        with patch("torch.save", side_effect=lambda payload, _: real_save(payload, buffer)), \
                patch.object(Path, "mkdir"):
            agent.save("never_written.pt", {"episode": 0})
        buffer.seek(0)
        payload = torch.load(buffer, map_location="cpu", weights_only=False)
        with patch("torch.load", return_value=payload):
            loaded = DoubleDQNAgent.load("never_written.pt", "cpu")
        self.assertEqual(loaded.config, agent.config)
        self.assertEqual(payload["extra"], {"episode": 0})
        for name in ("online_net", "target_net"):
            for x, y in zip(getattr(agent, name).parameters(), getattr(loaded, name).parameters()):
                torch.testing.assert_close(x, y)

    def test_cli_resume_preserves_saved_settings_without_training(self):
        agent = DoubleDQNAgent(replace(self.config, learning_rate=.002, reward_scale=.2,
                                       target_tau=None, target_update_interval=9))
        saved = TrainingConfig(episodes=3, epsilon_start=.4, epsilon_end=.2,
                                max_steps_per_episode=17, seed=8)
        payload = {"config": asdict(agent.config), "extra": {
            "episode": 3, "global_step": 51, "history": [],
            "training_config": asdict(saved), "training_state": {},
        }}
        args = ["train_doubledqn", "--resume", "source.pt", "--checkpoint", "new.pt",
                "--episodes", "5", "--device", "cpu", "--checkpoint-every", "2"]
        with patch("sys.argv", args), patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "exists", return_value=True), patch("torch.load", return_value=payload), \
                patch.object(DoubleDQNAgent, "load", return_value=agent), \
                patch.object(train_doubledqn, "train") as blocked_train:
            train_doubledqn.main()
        config, model = blocked_train.call_args.args
        self.assertEqual(config.episodes, 5)
        self.assertEqual(config.max_steps_per_episode, 17)
        self.assertEqual(config.epsilon_start, .4)
        self.assertEqual(config.checkpoint_every, 2)
        self.assertEqual(model.reward_scale, .2)
        self.assertIsNone(model.target_tau)
        self.assertEqual(model.target_update_interval, 9)

    def test_inference_cache_tracks_checkpoint_and_device(self):
        first, second, third = object(), object(), object()
        with patch.object(autoplay, "_agent", None), patch.object(autoplay, "_agent_key", None), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(DQNAgent, "load", side_effect=[first, second, third]) as load:
            self.assertIs(autoplay.load_policy("first.pt", "cpu"), first)
            self.assertIs(autoplay.load_policy("first.pt", "cpu"), first)
            self.assertIs(autoplay.load_policy("second.pt", "cpu"), second)
            self.assertIs(autoplay.load_policy("second.pt", None), third)
            self.assertEqual(load.call_count, 3)


if __name__ == "__main__":
    unittest.main()
