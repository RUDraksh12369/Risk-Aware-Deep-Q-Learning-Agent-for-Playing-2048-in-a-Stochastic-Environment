"""Neural Q-value model used by the 2048 DQN agent."""

from __future__ import annotations

from typing import Sequence
import math

import torch
from torch import nn


class DQN(nn.Module):
    """MLP that maps a log2-encoded 2048 board to one Q-value per move.

    The environment encodes tiles as exponents in ``[0, 16]``.  Normalising
    that fixed range inside the model keeps callers from needing a second,
    subtly different state-preprocessing path at training and inference.
    """

    def __init__(
        self,
        state_dim: int = 16,
        n_actions: int = 4,
        hidden_dims: Sequence[int] = (256, 256),
        max_exponent: float = 16.0,
        dueling: bool = True,
    ) -> None:
        super().__init__()
        if not math.isfinite(max_exponent) or max_exponent <= 0:
            raise ValueError("max_exponent must be positive and finite.")
        if state_dim <= 0 or n_actions <= 0:
            raise ValueError("state_dim and n_actions must be positive.")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths.")

        layers: list[nn.Module] = []
        in_features = state_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(in_features, width), nn.ReLU()))
            in_features = width
        self.dueling = dueling
        if dueling:
            self.feature_extractor = nn.Sequential(*layers)
            self.value_stream = nn.Linear(in_features, 1)
            self.advantage_stream = nn.Linear(in_features, n_actions)
        else:
            # Kept structurally identical to the original MLP so existing
            # checkpoints load without conversion.
            layers.append(nn.Linear(in_features, n_actions))
            self.network = nn.Sequential(*layers)
        self.state_dim = state_dim
        self.max_exponent = float(max_exponent)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, n_actions)`` Q-values for one or more states."""
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2:
            raise ValueError(f"Expected states shaped (batch, features), got {tuple(states.shape)}")
        if states.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected {self.state_dim} state features, got {states.shape[1]}."
            )
        normalized = states.float() / self.max_exponent
        if not self.dueling:
            return self.network(normalized)
        features = self.feature_extractor(normalized)
        advantages = self.advantage_stream(features)
        # Subtracting the mean makes the value/advantage decomposition unique.
        return self.value_stream(features) + advantages - advantages.mean(dim=1, keepdim=True)
