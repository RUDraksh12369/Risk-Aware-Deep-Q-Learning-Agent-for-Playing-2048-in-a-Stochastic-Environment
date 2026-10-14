"""Standalone Dueling-DQN fine-tuning and evaluation for the 2048 project.

This module deliberately does not alter the baseline DQN implementation.  It
can initialise a Dueling DQN from a baseline checkpoint, save its own
checkpoints, and run training/evaluation via ``python -m rl.duelingDQN``.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn

from game.environment import Env2048
from rl.replay_buffer import ReplayBuffer


@dataclass
class DuelingConfig:
    state_dim: int = 16
    n_actions: int = 4
    hidden_dims: tuple[int, ...] = (256, 256)
    max_exponent: float = 16.0
    gamma: float = 0.995
    learning_rate: float = 3e-4
    batch_size: int = 128
    replay_capacity: int = 100_000
    min_replay_size: int = 2_000
    target_update_interval: int = 1_000
    target_tau: Optional[float] = 0.005
    gradient_clip_norm: float = 10.0
    reward_scale: float = 0.01
    double_dqn: bool = True
    seed: Optional[int] = None
    device: Optional[str] = None


class DuelingDQN(nn.Module):
    """MLP Dueling DQN: ``Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)``."""

    def __init__(self, state_dim: int = 16, n_actions: int = 4,
                 hidden_dims: Sequence[int] = (256, 256), max_exponent: float = 16.0) -> None:
        super().__init__()
        if state_dim <= 0 or n_actions <= 0 or not hidden_dims:
            raise ValueError("state_dim, n_actions, and hidden_dims must be positive.")
        layers: list[nn.Module] = []
        input_width = state_dim
        for width in hidden_dims:
            if width <= 0:
                raise ValueError("hidden_dims must contain positive widths.")
            layers.extend((nn.Linear(input_width, width), nn.ReLU()))
            input_width = width
        self.feature_extractor = nn.Sequential(*layers)
        self.value_stream = nn.Linear(input_width, 1)
        self.advantage_stream = nn.Linear(input_width, n_actions)
        self.state_dim = state_dim
        self.max_exponent = float(max_exponent)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"Expected (batch, {self.state_dim}) states, got {tuple(states.shape)}.")
        features = self.feature_extractor(states.float() / self.max_exponent)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


def _checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {path} is not a dictionary checkpoint.")
    return payload


def _config_from_checkpoint(payload: dict[str, Any], device: str) -> DuelingConfig:
    saved = dict(payload.get("config", {}))
    allowed = set(DuelingConfig.__dataclass_fields__)
    values = {key: value for key, value in saved.items() if key in allowed}
    values["hidden_dims"] = tuple(values.get("hidden_dims", DuelingConfig.hidden_dims))
    values["device"] = device
    return DuelingConfig(**values)


def _feature_key_pairs(source: dict[str, torch.Tensor], destination: dict[str, torch.Tensor]) -> list[tuple[str, str]]:
    """Return compatible feature-layer pairs for dueling and legacy MLP keys."""
    pairs: list[tuple[str, str]] = []
    for dst_key, dst_value in destination.items():
        if not dst_key.startswith("feature_extractor."):
            continue
        candidates = [dst_key]
        # Old non-dueling network used the same alternating Linear/ReLU layout.
        candidates.append("network." + dst_key.split(".", 1)[1])
        for src_key in candidates:
            if src_key in source and source[src_key].shape == dst_value.shape:
                pairs.append((src_key, dst_key))
                break
    return pairs


class DuelingDQNAgent:
    """Independent replay/target-network agent used only by this module."""

    def __init__(self, config: Optional[DuelingConfig] = None) -> None:
        self.config = config or DuelingConfig()
        if self.config.min_replay_size < self.config.batch_size:
            raise ValueError("min_replay_size must be at least batch_size.")
        if self.config.replay_capacity < self.config.min_replay_size:
            raise ValueError("replay_capacity must be at least min_replay_size.")
        if self.config.seed is not None:
            random.seed(self.config.seed); np.random.seed(self.config.seed); torch.manual_seed(self.config.seed)
        self.device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        args = (self.config.state_dim, self.config.n_actions, self.config.hidden_dims, self.config.max_exponent)
        self.online_net = DuelingDQN(*args).to(self.device)
        self.target_net = DuelingDQN(*args).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(self.config.replay_capacity, seed=self.config.seed)
        self._rng = random.Random(self.config.seed)
        self.learn_steps = 0

    @torch.inference_mode()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        self.online_net.eval()
        return self.online_net(torch.as_tensor(state, dtype=torch.float32, device=self.device)).squeeze(0).cpu().numpy()

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float = 0.0) -> int:
        mask = np.asarray(legal_mask, dtype=bool)
        if not 0 <= epsilon <= 1 or mask.shape != (self.config.n_actions,) or not mask.any():
            raise ValueError("epsilon must be in [0, 1] and legal_mask must contain a legal action.")
        legal = np.flatnonzero(mask)
        if self._rng.random() < epsilon:
            return int(self._rng.choice(legal.tolist()))
        values = self.q_values(state)
        values[~mask] = -np.inf
        return int(np.argmax(values))

    def observe(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray,
                done: bool, next_legal_mask: np.ndarray) -> None:
        self.replay.add(state, action, reward * self.config.reward_scale, next_state, done, next_legal_mask)

    def learn(self) -> Optional[dict[str, float]]:
        if len(self.replay) < self.config.min_replay_size:
            return None
        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        next_mask = torch.as_tensor(batch.next_legal_masks, dtype=torch.bool, device=self.device)
        self.online_net.train()
        predicted = self.online_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            online_next = self.online_net(next_states).masked_fill(~next_mask, -torch.inf)
            if self.config.double_dqn:
                next_actions = online_next.argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).masked_fill(~next_mask, -torch.inf).max(dim=1).values
            next_q = torch.where(~dones.bool() & next_mask.any(dim=1), next_q, torch.zeros_like(next_q))
            targets = rewards + self.config.gamma * (1.0 - dones) * next_q
        loss = self.loss_fn(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True); loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step(); self.learn_steps += 1
        if self.config.target_tau is not None:
            with torch.no_grad():
                for target, online in zip(self.target_net.parameters(), self.online_net.parameters()):
                    target.lerp_(online, self.config.target_tau)
        elif self.learn_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return {"loss": float(loss.item()), "mean_q": float(predicted.mean().item()), "grad_norm": float(grad_norm)}

    def initialise_from_dqn(self, checkpoint_path: str | Path) -> dict[str, list[str]]:
        """Copy compatible DQN weights; never writes to the source checkpoint."""
        payload = _checkpoint(checkpoint_path, self.device)
        report: dict[str, list[str]] = {"online_transferred": [], "target_transferred": [], "initialized": []}
        for label, network in (("online", self.online_net), ("target", self.target_net)):
            source = payload.get(f"{label}_net", payload.get("online_net", {}))
            destination = network.state_dict()
            copied: list[str] = []
            for src_key, dst_key in _feature_key_pairs(source, destination):
                destination[dst_key] = source[src_key].detach().clone()
                copied.append(f"{src_key} -> {dst_key}")
            # Existing dueling checkpoints have compatible heads as well.
            for key in ("value_stream.weight", "value_stream.bias", "advantage_stream.weight", "advantage_stream.bias"):
                if key in source and source[key].shape == destination[key].shape:
                    destination[key] = source[key].detach().clone(); copied.append(f"{key} -> {key}")
                else:
                    report["initialized"].append(f"{label}.{key}")
            network.load_state_dict(destination)
            report[f"{label}_transferred"] = copied
        self.learn_steps = int(payload.get("learn_steps", 0))
        # Optimizer slots are valid only when every parameter was copied with identical layout.
        try:
            if "optimizer" in payload and not report["initialized"]:
                self.optimizer.load_state_dict(payload["optimizer"])
                report["optimizer"] = ["restored"]
            else:
                report["optimizer"] = ["initialized (incompatible or unavailable)"]
        except (ValueError, KeyError):
            report["optimizer"] = ["initialized (checkpoint optimizer incompatible)"]
        return report

    def save(self, path: str | Path, extra: Optional[dict[str, Any]] = None) -> None:
        destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "dueling_dqn_v1", "config": asdict(self.config),
                    "online_net": self.online_net.state_dict(), "target_net": self.target_net.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "replay": self.replay,
                    "learn_steps": self.learn_steps, "extra": extra or {}}, destination)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> tuple["DuelingDQNAgent", dict[str, Any]]:
        actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = _checkpoint(path, torch.device(actual_device))
        config = _config_from_checkpoint(payload, actual_device)
        agent = cls(config)
        agent.online_net.load_state_dict(payload["online_net"])
        agent.target_net.load_state_dict(payload.get("target_net", payload["online_net"]))
        if "optimizer" in payload: agent.optimizer.load_state_dict(payload["optimizer"])
        if isinstance(payload.get("replay"), ReplayBuffer): agent.replay = payload["replay"]
        agent.learn_steps = int(payload.get("learn_steps", 0)); agent.online_net.eval(); agent.target_net.eval()
        return agent, dict(payload.get("extra", {}))


def epsilon_by_step(step: int, start: float, end: float, decay_steps: int) -> float:
    return start + min(step / max(decay_steps, 1), 1.0) * (end - start)


def evaluate(agent: Any, episodes: int = 100, seed: int = 1_000, target_tile: int = 2048) -> dict[str, float]:
    scores: list[float] = []; tiles: list[float] = []; lengths: list[float] = []
    for offset in range(episodes):
        env = Env2048(seed=seed + offset); state = env.reset(); steps = 0
        while not env.done:
            mask = env.legal_action_mask()
            if not mask.any(): break
            state, _, done, _ = env.step(agent.select_action(state, mask, epsilon=0.0)); steps += 1
            if done: break
        scores.append(float(env.score)); tiles.append(float(env.game.max_tile())); lengths.append(float(steps))
    return {"episodes": float(episodes), "mean_score": float(np.mean(scores)), "best_score": float(np.max(scores)),
            "mean_max_tile": float(np.mean(tiles)), "mean_episode_length": float(np.mean(lengths)),
            "target_tile": float(target_tile), "target_tile_rate": float(np.mean(np.asarray(tiles) >= target_tile))}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def fine_tune(args: argparse.Namespace) -> list[dict[str, float]]:
    if args.resume:
        agent, extra = DuelingDQNAgent.load(args.resume, args.device)
        global_step = int(extra.get("global_step", 0)); start_episode = int(extra.get("episode", 0))
        history = list(extra.get("history", [])); transfer_report = extra.get("transfer_report", {})
    else:
        payload = _checkpoint(args.pretrained, torch.device(args.device or "cpu"))
        config = _config_from_checkpoint(payload, args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        config.seed = args.seed if args.seed is not None else config.seed
        agent = DuelingDQNAgent(config); transfer_report = agent.initialise_from_dqn(args.pretrained)
        global_step = int(payload.get("extra", {}).get("global_step", 0)); start_episode = int(payload.get("extra", {}).get("episode", 0)); history = []
    if args.seed is not None: random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    env = Env2048(seed=args.seed); checkpoint = Path(args.checkpoint); history_path = Path(args.history)
    for episode in range(start_episode + 1, start_episode + args.episodes + 1):
        state = env.reset(); losses: list[float] = []; reward_total = 0.0; steps = 0
        for _ in range(args.max_steps):
            mask = env.legal_action_mask()
            if not mask.any(): break
            epsilon = epsilon_by_step(global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
            action = agent.select_action(state, mask, epsilon); next_state, reward, done, _ = env.step(action)
            agent.observe(state, action, reward, next_state, done, np.zeros(env.n_actions, bool) if done else env.legal_action_mask())
            update = agent.learn()
            if update: losses.append(update["loss"])
            state = next_state; reward_total += reward; steps += 1; global_step += 1
            if done: break
        record = {"episode": float(episode), "steps": float(steps), "score": float(env.score), "max_tile": float(env.game.max_tile()), "reward": reward_total, "epsilon": epsilon_by_step(global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps), "loss": float(np.mean(losses)) if losses else float("nan")}
        history.append(record)
        if episode == start_episode + 1 or episode % 25 == 0: print("episode={episode:.0f} score={score:.0f} max_tile={max_tile:.0f} steps={steps:.0f} epsilon={epsilon:.3f} loss={loss:.4f}".format(**record))
        if episode % args.checkpoint_every == 0: agent.save(checkpoint, {"episode": episode, "global_step": global_step, "history": history, "transfer_report": transfer_report})
    agent.save(checkpoint, {"episode": start_episode + args.episodes, "global_step": global_step, "history": history, "transfer_report": transfer_report})
    _write_json(history_path, history); _write_json(history_path.with_name("transfer_report.json"), transfer_report)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="fine-tune from a DQN checkpoint or resume a Dueling checkpoint")
    train.add_argument("--pretrained", default="models/dqn_2048.pt"); train.add_argument("--resume")
    train.add_argument("--checkpoint", default="models/dueling_dqn_2048.pt"); train.add_argument("--history", default="artifacts/dueling_dqn/training_history.json")
    train.add_argument("--episodes", type=int, default=100); train.add_argument("--max-steps", type=int, default=5_000); train.add_argument("--checkpoint-every", type=int, default=25)
    train.add_argument("--epsilon-start", type=float, default=0.10); train.add_argument("--epsilon-end", type=float, default=0.05); train.add_argument("--epsilon-decay-steps", type=int, default=100_000)
    train.add_argument("--seed", type=int, default=42); train.add_argument("--device", default=None)
    ev = sub.add_parser("evaluate", help="evaluate a Dueling DQN checkpoint")
    ev.add_argument("--checkpoint", default="models/dueling_dqn_2048.pt"); ev.add_argument("--episodes", type=int, default=100); ev.add_argument("--seed", type=int, default=1_000); ev.add_argument("--target-tile", type=int, default=2048); ev.add_argument("--device", default=None); ev.add_argument("--output", default="artifacts/dueling_dqn/evaluation.json")
    compare = sub.add_parser("compare", help="evaluate baseline DQN and Dueling DQN with identical seeds")
    compare.add_argument("--dqn-checkpoint", default="models/dqn_2048.pt"); compare.add_argument("--dueling-checkpoint", default="models/dueling_dqn_2048.pt"); compare.add_argument("--episodes", type=int, default=100); compare.add_argument("--seed", type=int, default=1_000); compare.add_argument("--target-tile", type=int, default=2048); compare.add_argument("--device", default=None); compare.add_argument("--output", default="artifacts/dueling_dqn/comparison.json")
    args = parser.parse_args()
    if args.command == "train": fine_tune(args)
    elif args.command == "evaluate":
        agent, _ = DuelingDQNAgent.load(args.checkpoint, args.device); metrics = evaluate(agent, args.episodes, args.seed, args.target_tile); _write_json(Path(args.output), metrics); print(json.dumps(metrics, indent=2))
    else:
        from rl.agent import DQNAgent
        baseline = DQNAgent.load(args.dqn_checkpoint, args.device); dueling, _ = DuelingDQNAgent.load(args.dueling_checkpoint, args.device)
        results = {"dqn": evaluate(baseline, args.episodes, args.seed, args.target_tile), "dueling_dqn": evaluate(dueling, args.episodes, args.seed, args.target_tile)}
        _write_json(Path(args.output), results); print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
