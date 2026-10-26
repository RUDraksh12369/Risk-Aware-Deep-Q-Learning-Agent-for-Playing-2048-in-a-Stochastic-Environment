"""DQN agent: legal-action masking, experience replay, and target updates."""

from __future__ import annotations

import random
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn

from game.game import Game2048
from rl.dqn import DQN
from rl.replay_buffer import ReplayBuffer, TransitionBatch


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
    max_exponent: float = 16.0

    def validate(self) -> None:
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32):
            raise ValueError("seed must be an integer in [0, 2**32) or None.")
        for name in ("state_dim", "n_actions", "batch_size", "replay_capacity",
                     "min_replay_size", "target_update_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not self.hidden_dims or any(isinstance(w, bool) or not isinstance(w, int) or w < 1 for w in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")
        for name in ("learning_rate", "max_exponent", "gradient_clip_norm", "reward_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1].")
        if not self.batch_size <= self.min_replay_size <= self.replay_capacity:
            raise ValueError("require batch_size <= min_replay_size <= replay_capacity.")
        if self.target_tau is not None and not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must be in (0, 1] or None.")


class DQNAgent:
    """Standard (optionally Double) DQN implementation for :class:`Env2048`."""

    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        self.config = config or DQNConfig()
        self.config.validate()

        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

        requested = self.config.device
        self.device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_args = (self.config.state_dim, self.config.n_actions, self.config.hidden_dims)
        self.online_net = DQN(*model_args, dueling=self.config.dueling,
                              max_exponent=self.config.max_exponent).to(self.device)
        self.target_net = DQN(*model_args, dueling=self.config.dueling,
                              max_exponent=self.config.max_exponent).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(self.config.replay_capacity, seed=self.config.seed)
        self._rng = random.Random(self.config.seed)
        self.learn_steps = 0

    @torch.inference_mode()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        self._validate_state(state)
        self.online_net.eval()
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        values = self.online_net(tensor).squeeze(0).cpu().numpy()
        return values

    def _validate_state(self, state: np.ndarray) -> None:
        array = np.asarray(state)
        if array.shape != (self.config.state_dim,) or not np.isfinite(array).all():
            raise ValueError(f"state must be finite with shape ({self.config.state_dim},).")

    def _policy_q_values(self, state: np.ndarray, game: Optional[Game2048]) -> np.ndarray:
        """Action values for behaviour; learning uses the networks directly."""
        return self.q_values(state)

    def select_action(
        self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float = 0.0,
        *, game: Optional[Game2048] = None,
    ) -> int:
        """Choose an epsilon-greedy action, never selecting an illegal move."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        self._validate_state(state)
        mask = np.asarray(legal_mask, dtype=bool)
        if mask.shape != (self.config.n_actions,):
            raise ValueError(f"legal_mask must have shape ({self.config.n_actions},), got {mask.shape}")
        legal_actions = np.flatnonzero(mask)
        if legal_actions.size == 0:
            raise ValueError("Cannot select an action from an all-false legal mask.")
        if self._rng.random() < epsilon:
            return int(self._rng.choice(legal_actions.tolist()))
        values = self._policy_q_values(state, game)
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
        self._validate_state(state)
        self._validate_state(next_state)
        if np.asarray(next_legal_mask).shape != (self.config.n_actions,):
            raise ValueError(f"next_legal_mask must have shape ({self.config.n_actions},).")
        self.replay.add(
            state, action, reward * self.config.reward_scale, next_state, done, next_legal_mask
        )

    def compute_td_loss(self, batch: TransitionBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate loss, predictions and detached targets without updating weights.

        A time-limit truncation must retain done=False and its next legal mask;
        only environment termination suppresses the bootstrap.
        """
        count = len(batch.states)
        expected = {"states": (count, self.config.state_dim),
                    "next_states": (count, self.config.state_dim),
                    "actions": (count,), "rewards": (count,), "dones": (count,),
                    "next_legal_masks": (count, self.config.n_actions)}
        for name, shape in expected.items():
            array = np.asarray(getattr(batch, name))
            if count == 0 or array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"batch.{name} must be finite with shape {shape}.")
        if not np.issubdtype(batch.actions.dtype, np.integer) or np.any(batch.actions < 0) or np.any(batch.actions >= self.config.n_actions):
            raise ValueError("batch.actions must contain integer action indices.")
        if not np.isin(batch.dones, [0, 1]).all() or not np.isin(batch.next_legal_masks, [0, 1]).all():
            raise ValueError("Batch dones and legal masks must be boolean.")
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        next_mask = torch.as_tensor(batch.next_legal_masks, dtype=torch.bool, device=self.device)

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

        return self.loss_fn(predicted_q, targets), predicted_q, targets

    def learn(self) -> Optional[dict[str, float]]:
        """Perform one replay update, or return ``None`` during warm-up."""
        if len(self.replay) < self.config.min_replay_size:
            return None
        self.online_net.train()
        loss, predicted_q, targets = self.compute_td_loss(self.replay.sample(self.config.batch_size))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()

        self.learn_steps += 1
        self.update_target_network()
        return {"loss": float(loss.item()), "mean_q": float(predicted_q.mean().item()),
                "grad_norm": float(grad_norm),
                "mean_target": float(targets.mean().item()),
                "mean_abs_td_error": float((targets - predicted_q.detach()).abs().mean().item())}

    def update_target_network(self) -> None:
        """Synchronize after an optimizer update, using the configured strategy."""
        if self.config.target_tau is not None:
            with torch.no_grad():
                for target_parameter, online_parameter in zip(
                    self.target_net.parameters(), self.online_net.parameters()
                ):
                    target_parameter.lerp_(online_parameter, self.config.target_tau)
        elif self.learn_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def _checkpoint_metadata(self) -> dict[str, Any]:
        """Additional policy settings; baseline checkpoint format stays valid."""
        return {}

    @classmethod
    def _from_checkpoint(cls, config: DQNConfig, checkpoint: dict[str, Any]) -> "DQNAgent":
        return cls(config)

    def save(self, path: str | Path, extra: Optional[dict[str, Any]] = None) -> None:
        """Write a portable checkpoint containing model and optimizer state."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            **self._checkpoint_metadata(),
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
        if checkpoint.get("format") == "dueling_dqn_v1":
            # The standalone trainer always uses dueling heads, but does not
            # store the shared agent's architecture flag.
            saved_config["dueling"] = True
        saved_config["hidden_dims"] = tuple(saved_config["hidden_dims"])
        # Checkpoints produced by the original MLP do not have the enhanced
        # fields. Preserve their architecture and reward scale on load.
        saved_config.setdefault("dueling", False)
        saved_config.setdefault("reward_scale", 1.0)
        saved_config.setdefault("target_tau", None)
        saved_config["device"] = str(map_location)
        agent_class = cls
        agent_type = checkpoint.get("agent_type")
        if agent_type == "ra_d3qn":
            from rl.risk_aware import RiskAwareDQNAgent

            if cls not in (DQNAgent, RiskAwareDQNAgent):
                raise ValueError("Use DQNAgent.load or RiskAwareDQNAgent.load for an RA-D3QN checkpoint.")
            agent_class = RiskAwareDQNAgent
        elif agent_type is not None:
            raise ValueError(f"Unknown checkpoint agent_type: {agent_type}")
        agent = agent_class._from_checkpoint(DQNConfig(**saved_config), checkpoint)
        agent.online_net.load_state_dict(checkpoint["online_net"])
        agent.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["online_net"]))
        if "optimizer" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.learn_steps = int(checkpoint.get("learn_steps", 0))
        agent.online_net.eval()
        agent.target_net.eval()
        return agent
