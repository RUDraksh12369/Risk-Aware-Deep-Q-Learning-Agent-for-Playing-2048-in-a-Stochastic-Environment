"""
game/game.py

Pure 2048 game mechanics. No RL, no rendering, no NumPy dependency required
for correctness (but we use it for convenience). This module is the single
source of truth for board rules: movement, merging, tile spawning, and
game-over detection.

Design notes (see project context, Section 6 & 9):
- Board is a 4x4 grid of ints (0 = empty, otherwise the tile's face value:
  2, 4, 8, ... 2048, ...).
- All four moves (UP, DOWN, LEFT, RIGHT) are implemented via a single
  "compress + merge" primitive applied to rows, using array reversals /
  transposes to reuse the same core logic for every direction.
- This class intentionally exposes a `clone()` method so that other
  modules (risk estimation, Expectimax, action simulation) can explore
  hypothetical moves WITHOUT mutating the real game state.
"""

from __future__ import annotations

import copy
import random
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

BOARD_SIZE = 4


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_NAMES = {
    Action.UP: "UP",
    Action.DOWN: "DOWN",
    Action.LEFT: "LEFT",
    Action.RIGHT: "RIGHT",
}


class Game2048:
    """Pure game-mechanics engine for 2048."""

    def __init__(
        self,
        size: int = BOARD_SIZE,
        p_spawn_2: float = 0.9,
        seed: Optional[int] = None,
    ):
        """
        Args:
            size: board dimension (default 4x4).
            p_spawn_2: probability a spawned tile is a 2 (rest is a 4).
                       Exposed here (not hard-coded) so Section 25's
                       generalization/robustness experiments can alter the
                       stochastic distribution, e.g. p_spawn_2=0.7.
            seed: optional int for a dedicated RNG instance (keeps
                  experiments reproducible without touching global random
                  state).
        """
        self.size = size
        self.p_spawn_2 = p_spawn_2
        self._rng = random.Random(seed)

        self.board: np.ndarray = np.zeros((size, size), dtype=np.int64)
        self.score: int = 0
        self.moves_made: int = 0
        self.done: bool = False

        self.reset()

    # ------------------------------------------------------------------ #
    # Setup / reset
    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        """Start a fresh game: empty board + two random tiles."""
        self.board = np.zeros((self.size, self.size), dtype=np.int64)
        self.score = 0
        self.moves_made = 0
        self.done = False
        self._spawn_tile()
        self._spawn_tile()
        return self.board.copy()

    def _spawn_tile(self) -> bool:
        """Place a new tile (2 w.p. p_spawn_2, else 4) in a random empty cell.

        Returns False if there was no empty cell to spawn into.
        """
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            return False
        r, c = self._rng.choice(empty)
        value = 2 if self._rng.random() < self.p_spawn_2 else 4
        self.board[r, c] = value
        return True

    # ------------------------------------------------------------------ #
    # Core move primitive: everything reduces to "compress a row left"
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compress_and_merge_row(row: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        """Slide a single row left, merging equal adjacent tiles once each.

        Returns (new_row, score_gained, changed_flag).
        """
        size = len(row)
        nonzero = row[row != 0]

        merged: List[int] = []
        gained = 0
        i = 0
        while i < len(nonzero):
            if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
                merged_value = int(nonzero[i]) * 2
                merged.append(merged_value)
                gained += merged_value
                i += 2
            else:
                merged.append(int(nonzero[i]))
                i += 1

        new_row = np.array(merged + [0] * (size - len(merged)), dtype=np.int64)
        changed = not np.array_equal(new_row, row)
        return new_row, gained, changed

    def _move_left(self, board: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        new_board = np.zeros_like(board)
        total_gain = 0
        any_changed = False
        for r in range(board.shape[0]):
            new_row, gained, changed = self._compress_and_merge_row(board[r])
            new_board[r] = new_row
            total_gain += gained
            any_changed = any_changed or changed
        return new_board, total_gain, any_changed

    def _apply_move(self, board: np.ndarray, action: Action) -> Tuple[np.ndarray, int, bool]:
        """Apply `action` to `board` and return (new_board, score_gained, changed).

        Implemented by rotating/flipping the board so every direction reuses
        the single `_move_left` primitive, then rotating back.
        """
        if action == Action.LEFT:
            new_board, gained, changed = self._move_left(board)
        elif action == Action.RIGHT:
            flipped = np.fliplr(board)
            moved, gained, changed = self._move_left(flipped)
            new_board = np.fliplr(moved)
        elif action == Action.UP:
            rotated = board.T
            moved, gained, changed = self._move_left(rotated)
            new_board = moved.T
        elif action == Action.DOWN:
            rotated = np.fliplr(board.T)
            moved, gained, changed = self._move_left(rotated)
            new_board = np.fliplr(moved).T
        else:
            raise ValueError(f"Unknown action: {action}")

        return new_board, gained, changed

    # ------------------------------------------------------------------ #
    # Public step API (mutates real game state)
    # ------------------------------------------------------------------ #
    def step(self, action: Action) -> Tuple[np.ndarray, int, bool, dict]:
        """Execute `action` on the real board.

        Returns (next_board, reward, done, info) — a Gym-style tuple.
        `reward` here is the raw game score gained this step (merges only).
        Reward SHAPING (risk penalties etc.) is intentionally NOT applied
        here; that belongs to the RL environment wrapper (environment.py)
        so this class stays a pure, reusable rules engine.

        info dict contains:
            - "valid": whether the move actually changed the board
            - "spawned": whether a new tile was successfully spawned
        """
        if self.done:
            return self.board.copy(), 0, True, {"valid": False, "spawned": False}

        new_board, gained, changed = self._apply_move(self.board, action)

        info = {"valid": changed, "spawned": False}

        if changed:
            self.board = new_board
            self.score += gained
            self.moves_made += 1
            info["spawned"] = self._spawn_tile()

        self.done = self._is_game_over()
        return self.board.copy(), gained, self.done, info

    # ------------------------------------------------------------------ #
    # Simulation (non-mutating) — required for action-specific risk (Sec 11)
    # and for Expectimax / heuristic search baselines.
    # ------------------------------------------------------------------ #
    def simulate_move(self, action: Action, board: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, int, bool]:
        """Return the resulting board/score/validity of `action` WITHOUT
        mutating self.board and WITHOUT spawning a random tile.

        This is the "deterministic afterstate" — useful for risk estimation
        and Expectimax, where the stochastic tile spawn is handled
        separately (Expectimax enumerates spawn possibilities explicitly;
        the risk module evaluates the afterstate directly).
        """
        source = self.board if board is None else board
        new_board, gained, changed = self._apply_move(source, action)
        return new_board, gained, changed

    def legal_actions(self, board: Optional[np.ndarray] = None) -> List[Action]:
        """Which of the 4 actions actually change the board."""
        source = self.board if board is None else board
        legal = []
        for action in Action:
            _, _, changed = self._apply_move(source, action)
            if changed:
                legal.append(action)
        return legal

    def _is_game_over(self) -> bool:
        """No legal actions remain (board full AND no merges possible)."""
        return len(self.legal_actions()) == 0

    # ------------------------------------------------------------------ #
    # Convenience / introspection
    # ------------------------------------------------------------------ #
    def clone(self) -> "Game2048":
        """Deep copy of the current game (independent RNG state included).

        Used by search-based baselines (Expectimax) and by any module that
        needs to roll forward multiple hypothetical moves.
        """
        new_game = Game2048.__new__(Game2048)
        new_game.size = self.size
        new_game.p_spawn_2 = self.p_spawn_2
        new_game._rng = random.Random()
        new_game._rng.setstate(self._rng.getstate())
        new_game.board = self.board.copy()
        new_game.score = self.score
        new_game.moves_made = self.moves_made
        new_game.done = self.done
        return new_game

    def max_tile(self) -> int:
        return int(self.board.max())

    def empty_cells(self) -> int:
        return int(np.sum(self.board == 0))

    def __str__(self) -> str:
        lines = []
        for row in self.board:
            lines.append(" ".join(f"{v:5d}" if v else f"{'.':>5}" for v in row))
        return "\n".join(lines) + f"\nScore: {self.score}  Moves: {self.moves_made}  Done: {self.done}"


if __name__ == "__main__":
    # Quick manual smoke test.
    g = Game2048(seed=42)
    print("Initial board:")
    print(g)
    print()

    for a in [Action.LEFT, Action.UP, Action.RIGHT, Action.DOWN]:
        board, reward, done, info = g.step(a)
        print(f"Action: {ACTION_NAMES[a]:<6} valid={info['valid']} reward={reward} done={done}")
        print(g)
        print()
        if done:
            break
