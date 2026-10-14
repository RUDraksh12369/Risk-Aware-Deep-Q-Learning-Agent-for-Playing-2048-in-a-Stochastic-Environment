"""DQN agent: legal-action masking, experience replay, and target updates."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn

from rl.dqn import DQN
from rl.replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    state_dim: int = 16
    n_actions: int = 4
    hidden_dims: tuple[int, ...] = (256, 256)
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
    dueling: bool = True
    seed: Optional[int] = None
    device: Optional[str] = None


class DQNAgent:
    """Standard (optionally Double) DQN implementation for :class:`Env2048`."""

    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        self.config = config or DQNConfig()
        if not 0.0 <= self.config.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if self.config.min_replay_size < self.config.batch_size:
            raise ValueError("min_replay_size must be at least batch_size.")
        if self.config.replay_capacity < self.config.min_replay_size:
            raise ValueError("replay_capacity must be at least min_replay_size.")
        if self.config.target_update_interval < 1:
            raise ValueError("target_update_interval must be positive.")
        if self.config.target_tau is not None and not 0.0 < self.config.target_tau <= 1.0:
            raise ValueError("target_tau must be in (0, 1] or None.")
        if self.config.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")
        if self.config.reward_scale <= 0:
            raise ValueError("reward_scale must be positive.")

        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

        requested = self.config.device
        self.device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_args = (self.config.state_dim, self.config.n_actions, self.config.hidden_dims)
        self.online_net = DQN(*model_args, dueling=self.config.dueling).to(self.device)
        self.target_net = DQN(*model_args, dueling=self.config.dueling).to(self.device)
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
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        values = self.online_net(tensor).squeeze(0).cpu().numpy()
        return values

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float = 0.0) -> int:
        """Choose an epsilon-greedy action, never selecting an illegal move."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        mask = np.asarray(legal_mask, dtype=bool)
        if mask.shape != (self.config.n_actions,):
            raise ValueError(f"legal_mask must have shape ({self.config.n_actions},), got {mask.shape}")
        legal_actions = np.flatnonzero(mask)
        if legal_actions.size == 0:
            raise ValueError("Cannot select an action from an all-false legal mask.")
        if self._rng.random() < epsilon:
            return int(self._rng.choice(legal_actions.tolist()))
        values = self.q_values(state)
        values[~mask] = -np.inf
        return int(np.argmax(values))

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_mask: np.ndarray,
    ) -> None:
        # Scores in 2048 grow much faster than the bounded board encoding.
        # Scale only the learning signal; public episode metrics stay in raw
        # game-score units for direct comparison.
        self.replay.add(
            state, action, reward * self.config.reward_scale, next_state, done, next_legal_mask
        )

    def learn(self) -> Optional[dict[str, float]]:
        """Perform one replay update, or return ``None`` during warm-up."""
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
        predicted_q = self.online_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            if self.config.double_dqn:
                # Keep the existing config/checkpoint interface compatible
                # while the algorithm lives in its dedicated module.
                from rl.doubledqn import double_dqn_next_q

                next_q = double_dqn_next_q(self.online_net, self.target_net, next_states, next_mask)
            else:
                # Standard DQN both selects and evaluates with the target network.
                next_q = self.target_net(next_states).masked_fill(~next_mask, -torch.inf).max(dim=1).values
            # A terminal state normally has no legal action. Treat an empty
            # mask defensively the same way, so malformed/off-policy data
            # cannot introduce an infinite Bellman target.
            can_bootstrap = ~dones.bool() & next_mask.any(dim=1)
            next_q = torch.where(can_bootstrap, next_q, torch.zeros_like(next_q))
            targets = rewards + self.config.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(predicted_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()

        self.learn_steps += 1
        if self.config.target_tau is not None:
            with torch.no_grad():
                for target_parameter, online_parameter in zip(
                    self.target_net.parameters(), self.online_net.parameters()
                ):
                    target_parameter.lerp_(online_parameter, self.config.target_tau)
        elif self.learn_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return {"loss": float(loss.item()), "mean_q": float(predicted_q.mean().item()), "grad_norm": float(grad_norm)}

    def save(self, path: str | Path, extra: Optional[dict[str, Any]] = None) -> None:
        """Write a portable checkpoint containing model and optimizer state."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": asdict(self.config),
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learn_steps": self.learn_steps,
            "extra": extra or {},
        }, destination)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "DQNAgent":
        map_location = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
        saved_config = dict(checkpoint["config"])
        saved_config["hidden_dims"] = tuple(saved_config["hidden_dims"])
        # Checkpoints produced by the original MLP do not have the enhanced
        # fields. Preserve their architecture and reward scale on load.
        saved_config.setdefault("dueling", False)
        saved_config.setdefault("reward_scale", 1.0)
        saved_config.setdefault("target_tau", None)
        saved_config["device"] = str(map_location)
        agent = cls(DQNConfig(**saved_config))
        agent.online_net.load_state_dict(checkpoint["online_net"])
        agent.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["online_net"]))
        if "optimizer" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.learn_steps = int(checkpoint.get("learn_steps", 0))
        agent.online_net.eval()
        agent.target_net.eval()
        return agent
