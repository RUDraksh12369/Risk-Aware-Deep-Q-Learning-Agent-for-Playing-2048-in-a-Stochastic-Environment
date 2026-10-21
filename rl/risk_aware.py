"""Policy-only Risk-Aware Dueling Double DQN (RA-D3QN)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Optional

import numpy as np

from game.environment import encode_board
from game.game import Action, Game2048
from rl.agent import DQNConfig
from rl.doubledqn import DoubleDQNAgent
from rl.risk import RiskConfig, risk_adjusted_q_values


@dataclass(frozen=True)
class RiskAwareConfig:
    """Risk penalty in learned Q units; defaults require experimental tuning."""

    risk_coefficient: float = 0.1
    weights: RiskConfig = field(default_factory=RiskConfig)

    def __post_init__(self) -> None:
        if not math.isfinite(self.risk_coefficient) or self.risk_coefficient < 0:
            raise ValueError("risk_coefficient must be finite and nonnegative.")
        if not isinstance(self.weights, RiskConfig):
            raise ValueError("weights must be a RiskConfig.")


class RiskAwareDQNAgent(DoubleDQNAgent):
    """Use Q - lambda*R for exploitation, uniform legal exploration otherwise.

    Risk changes the behaviour policy only. Replay rewards and Double DQN
    targets remain unchanged: Q estimates the ordinary score-return objective,
    not a risk-adjusted return. Lambda zero recovers Dueling Double DQN.
    """

    def __init__(
        self, config: Optional[DQNConfig] = None, risk_config: Optional[RiskAwareConfig] = None
    ) -> None:
        self.risk_config = risk_config or RiskAwareConfig()
        super().__init__(replace(config or DQNConfig(), dueling=True, double_dqn=True))

    def select_action(
        self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float = 0.0,
        *, game: Optional[Game2048] = None,
    ) -> int:
        if game is not None:
            legal = game.legal_actions()
            expected_mask = np.array([action in legal for action in Action])
            if not np.array_equal(np.asarray(legal_mask, dtype=bool), expected_mask):
                raise ValueError("legal_mask must match the current game board.")
        return super().select_action(state, legal_mask, epsilon, game=game)

    def _policy_q_values(self, state: np.ndarray, game: Optional[Game2048]) -> np.ndarray:
        if self.risk_config.risk_coefficient == 0:
            return self.q_values(state)
        if game is None:
            raise ValueError("Risk-aware exploitation requires game= with the current raw board.")
        if not np.array_equal(np.asarray(state), encode_board(game.board)):
            raise ValueError("state must encode the current game board.")
        return risk_adjusted_q_values(
            game, self.q_values(state), self.risk_config.risk_coefficient, self.risk_config.weights
        )

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {"agent_type": "ra_d3qn", "risk_config": asdict(self.risk_config)}

    @classmethod
    def _from_checkpoint(cls, config: DQNConfig, checkpoint: dict[str, Any]) -> "RiskAwareDQNAgent":
        if checkpoint.get("agent_type") != "ra_d3qn":
            raise ValueError("Expected an RA-D3QN checkpoint with saved risk settings.")
        if not config.dueling or not config.double_dqn:
            raise ValueError("RA-D3QN checkpoints must enable dueling and double_dqn.")
        saved_risk = dict(checkpoint["risk_config"])
        saved_risk["weights"] = RiskConfig(**saved_risk["weights"])
        return cls(config, RiskAwareConfig(**saved_risk))
