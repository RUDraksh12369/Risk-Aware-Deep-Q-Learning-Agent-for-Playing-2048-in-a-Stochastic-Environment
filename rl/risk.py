"""Board/afterstate risk and Q - lambda * risk action evaluation for 2048.

The composite score is a configurable heuristic, not a calibrated probability
of eventual failure. Only the immediate next-spawn game-over component is an
exact probability. No live board or random-number generator is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from game.game import Action, Game2048


@dataclass(frozen=True)
class RiskConfig:
    """Nonnegative relative weights; equal defaults are an unvalidated choice."""

    mobility_weight: float = 1.0
    crowding_weight: float = 1.0
    game_over_weight: float = 1.0

    def __post_init__(self) -> None:
        weights = (self.mobility_weight, self.crowding_weight, self.game_over_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Risk weights must be finite and nonnegative.")
        if not math.isfinite(sum(weights)) or sum(weights) <= 0:
            raise ValueError("Risk weights must have a finite, positive sum.")


@dataclass(frozen=True)
class BoardRisk:
    """Interpretable measurements on a raw board or deterministic afterstate."""

    legal_moves: int
    empty_cells: int
    mobility_risk: float
    crowding_risk: float
    spawn_game_over_probability: float
    score: float


def board_risk(
    game: Game2048, board: np.ndarray | None = None, config: RiskConfig | None = None
) -> BoardRisk:
    """Measure mobility, crowding, and immediate spawn danger in [0, 1].

    Mobility and crowding refer to the supplied board *before* a spawn. With
    two or more empty cells, one spawn cannot end a reachable game. With one
    empty cell, check the two possible tile values using the existing rules.
    Full boards retain their current terminal status. This is a bounded local
    calculation, not a rollout or multi-step search.
    """
    config = config or RiskConfig()
    source = np.asarray(game.board if board is None else board)
    if game.size < 2 or source.shape != (game.size, game.size):
        raise ValueError("board must match the game's square board, of size at least 2.")
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError("board must contain raw integer tile values, not encoded states.")
    if np.any(source < 0) or np.any(source == 1) or np.any(source & (source - 1)):
        raise ValueError("Tiles must be zero or powers of two greater than or equal to 2.")
    if not math.isfinite(game.p_spawn_2) or not 0.0 <= game.p_spawn_2 <= 1.0:
        raise ValueError("p_spawn_2 must be finite and in [0, 1].")

    legal_moves = len(game.legal_actions(source))
    empty_cells = int(np.count_nonzero(source == 0))
    mobility_risk = 1.0 - legal_moves / len(Action)
    crowding_risk = 1.0 - empty_cells / source.size
    spawn_game_over_probability = float(empty_cells == 0 and legal_moves == 0)
    if legal_moves and empty_cells == 1:
        row, col = np.argwhere(source == 0)[0]
        candidate = source.copy()
        for tile, probability in ((2, game.p_spawn_2), (4, 1.0 - game.p_spawn_2)):
            candidate[row, col] = tile
            if not game.legal_actions(candidate):
                spawn_game_over_probability += probability

    total_weight = config.mobility_weight + config.crowding_weight + config.game_over_weight
    score = (
        config.mobility_weight * mobility_risk
        + config.crowding_weight * crowding_risk
        + config.game_over_weight * spawn_game_over_probability
    ) / total_weight
    return BoardRisk(
        legal_moves, empty_cells, mobility_risk, crowding_risk,
        spawn_game_over_probability, float(score),
    )


def action_risks(game: Game2048, config: RiskConfig | None = None) -> np.ndarray:
    """Return R(s, a) in Action order; illegal actions have positive infinity.

    Evaluate each deterministic afterstate, before the actual random spawn.
    The spawn-danger component uses the game's configured tile probabilities.
    """
    risks = np.full(len(Action), np.inf, dtype=np.float64)
    for action in Action:
        afterstate, _, valid = game.simulate_move(action)
        if valid:
            risks[int(action)] = board_risk(game, afterstate, config).score
    return risks


def risk_adjusted_q_values(
    game: Game2048,
    q_values: np.ndarray,
    risk_coefficient: float,
    config: RiskConfig | None = None,
) -> np.ndarray:
    """Return masked Q(s, a) - lambda * R(s, a), leaving input Q values intact.

    Lambda is measured in the network's Q-value units (including any existing
    reward scaling). Lambda zero reproduces the masked baseline values. This
    utility does not alter rewards, replay, or Bellman targets.
    """
    if not math.isfinite(risk_coefficient) or risk_coefficient < 0:
        raise ValueError("risk_coefficient must be finite and nonnegative.")
    values = np.asarray(q_values, dtype=np.float64)
    if values.shape != (len(Action),) or not np.isfinite(values).all():
        raise ValueError("q_values must contain four finite action values.")
    risks = action_risks(game, config)
    legal = np.isfinite(risks)
    if not legal.any():
        raise ValueError("Cannot evaluate actions on a board with no legal moves.")
    adjusted = np.full(len(Action), -np.inf, dtype=np.float64)
    # Index legal actions first: lambda=0 must never multiply infinity.
    adjusted[legal] = values[legal] - risk_coefficient * risks[legal]
    return adjusted
