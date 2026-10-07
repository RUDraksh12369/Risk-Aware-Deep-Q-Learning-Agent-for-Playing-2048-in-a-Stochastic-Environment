"""Fixed-size experience replay storage for DQN."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_legal_mask: np.ndarray


@dataclass(frozen=True)
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    next_legal_masks: np.ndarray


class ReplayBuffer:
    """Uniform replay buffer with defensive copies at its public boundary."""

    def __init__(self, capacity: int = 100_000, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self._items: list[Transition] = []
        self._position = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_mask: np.ndarray,
    ) -> None:
        transition = Transition(
            state=np.asarray(state, dtype=np.float32).copy(),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32).copy(),
            done=bool(done),
            next_legal_mask=np.asarray(next_legal_mask, dtype=bool).copy(),
        )
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._position] = transition
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> TransitionBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(self) < batch_size:
            raise ValueError(f"Need {batch_size} transitions, only have {len(self)}.")
        samples: Sequence[Transition] = self._rng.sample(self._items, batch_size)
        return TransitionBatch(
            states=np.stack([item.state for item in samples]),
            actions=np.asarray([item.action for item in samples], dtype=np.int64),
            rewards=np.asarray([item.reward for item in samples], dtype=np.float32),
            next_states=np.stack([item.next_state for item in samples]),
            dones=np.asarray([item.done for item in samples], dtype=np.float32),
            next_legal_masks=np.stack([item.next_legal_mask for item in samples]),
        )
