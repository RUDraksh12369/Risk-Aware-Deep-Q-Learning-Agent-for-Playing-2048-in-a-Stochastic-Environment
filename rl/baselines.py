"""Non-learning reference policies sharing the evaluator's action interface."""

from __future__ import annotations

import random

import numpy as np

from game.game import Action, Game2048


def _legal_actions(legal_mask: np.ndarray, game: Game2048 | None) -> list[int]:
    mask = np.asarray(legal_mask, dtype=bool)
    if mask.shape != (len(Action),):
        raise ValueError("legal_mask must contain four action flags.")
    if game is not None:
        actual = game.legal_actions()
        if not np.array_equal(mask, [action in actual for action in Action]):
            raise ValueError("legal_mask must match the current game board.")
    legal = np.flatnonzero(mask).tolist()
    if not legal:
        raise ValueError("Cannot select an action from an all-false legal mask.")
    return legal


class RandomAgent:
    """Uniform legal-action policy with its own reproducible RNG."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(
        self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float = 0.0,
        *, game: Game2048 | None = None,
    ) -> int:
        # Epsilon/state are accepted for evaluator compatibility; no Q model exists.
        return int(self._rng.choice(_legal_actions(legal_mask, game)))


class GreedyAgent:
    """Maximize the immediate merge score; ties use the lowest action id.

    Move simulation is deterministic and never spawns tiles or changes the
    live game. This is an immediate-score baseline, not the GUI heuristic.
    """

    def select_action(
        self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float = 0.0,
        *, game: Game2048 | None = None,
    ) -> int:
        if game is None:
            raise ValueError("GreedyAgent requires game= with the current raw board.")
        legal = _legal_actions(legal_mask, game)
        return int(max(legal, key=lambda action: game.simulate_move(Action(action))[1]))
