"""Optional GUI policy with legal masking and inference-only tie-breaking.

Select it with ``--agent inference.optimized_autoplay:choose_action``. The
trained model stays unchanged. Environment variables configure the relative
Q-value tie window and feature weights.
"""

from __future__ import annotations

import os
import math

from game.game import Action, Game2048
from inference.autoplay import load_policy
from inference.heuristics import HeuristicWeights, choose_tiebroken_action
from rl.risk_aware import RiskAwareDQNAgent


def _float_setting(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def choose_action(game: Game2048) -> Action:
    """Greedy legal action with optional close-Q heuristic tie-breaking."""
    weights = HeuristicWeights(
        monotonicity=_float_setting("DQN_HEURISTIC_MONOTONICITY", 1.0),
        smoothness=_float_setting("DQN_HEURISTIC_SMOOTHNESS", 1.0),
        empty_cells=_float_setting("DQN_HEURISTIC_EMPTY_CELLS", 1.0),
        highest_tile_corner=_float_setting("DQN_HEURISTIC_CORNER", 1.0),
    )
    agent = load_policy(os.environ.get("DQN_CHECKPOINT"))
    risk_coefficient = (
        _float_setting("DQN_RISK_COEFFICIENT", 0.2)
        if isinstance(agent, RiskAwareDQNAgent) else None
    )
    return choose_tiebroken_action(
        agent,
        game,
        tie_tolerance=_float_setting("DQN_TIE_TOLERANCE", 0.01),
        weights=weights,
        risk_coefficient=risk_coefficient,
    )
