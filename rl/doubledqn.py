"""Double DQN using the existing DQN networks, replay, and learning loop."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch
from torch import nn

from rl.agent import DQNAgent, DQNConfig


@torch.no_grad()
def double_dqn_next_q(
    online_net: nn.Module,
    target_net: nn.Module,
    next_states: torch.Tensor,
    next_mask: torch.Tensor,
) -> torch.Tensor:
    """Select the best legal action online and evaluate it with the target.

    The shared learning loop suppresses bootstrapping for terminal states
    and empty legal-action masks before constructing the Bellman targets.
    """
    online_next_q = online_net(next_states).masked_fill(~next_mask, -torch.inf)
    next_actions = online_next_q.argmax(dim=1, keepdim=True)
    return target_net(next_states).gather(1, next_actions).squeeze(1)


class DoubleDQNAgent(DQNAgent):
    """DQN agent that always trains with Double DQN targets.

    Construction initializes fresh networks and replay through DQNAgent.
    The inherited save/load interface remains compatible with evaluation
    and inference, and configuration supplied by the caller is not mutated.
    """

    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        super().__init__(replace(config or DQNConfig(), double_dqn=True))
