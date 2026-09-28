"""Monte Carlo tree search agent for 2048.

The agent is deliberately independent of the project's reinforcement-learning
modules. It treats 2048 as a stochastic planning problem: action nodes choose a
slide direction, then the simulator samples the random tile spawn.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from game.game import Action, Game2048

BoardKey = Tuple[int, ...]


@dataclass
class MCTSConfig:
    simulations: int = 160
    rollout_depth: int = 36
    exploration: float = 1.4
    time_limit_ms: Optional[int] = None
    p_spawn_2: float = 0.9
    seed: Optional[int] = None


@dataclass
class ActionStats:
    visits: int = 0
    value: float = 0.0


@dataclass
class Node:
    board: np.ndarray
    visits: int = 0
    value: float = 0.0
    children: Dict[Action, Dict[BoardKey, "Node"]] = field(default_factory=dict)
    action_stats: Dict[Action, ActionStats] = field(default_factory=dict)


class MCTS2048Agent:
    """Callable MCTS player for ``Game2048``.

    Each call builds a fresh search tree from the current board and returns the
    action with the best average simulated return.
    """

    def __init__(self, config: Optional[MCTSConfig] = None):
        self.config = config or MCTSConfig()
        self._rng = random.Random(self.config.seed)
        self._rules = Game2048(p_spawn_2=self.config.p_spawn_2, seed=self.config.seed)

    def __call__(self, game: Game2048) -> Action:
        return self.choose_action(game)

    def choose_action(self, game: Game2048) -> Action:
        legal = game.legal_actions()
        if not legal:
            raise ValueError("MCTS cannot choose an action from a terminal board.")

        root = Node(game.board.copy())
        deadline = None
        if self.config.time_limit_ms is not None:
            deadline = time.perf_counter() + self.config.time_limit_ms / 1000.0

        completed = 0
        while completed < self.config.simulations:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            self._simulate(root)
            completed += 1

        return max(
            legal,
            key=lambda action: (
                self._average_value(root.action_stats.get(action)),
                root.action_stats.get(action, ActionStats()).visits,
            ),
        )

    def _simulate(self, root: Node) -> float:
        node = root
        path: list[tuple[Node, Optional[Action]]] = []
        total_reward = 0.0
        depth = 0

        while depth < self.config.rollout_depth:
            legal = self._legal_actions(node.board)
            if not legal:
                break

            action = self._select_action(node, legal)
            afterstate, gained, changed = self._rules.simulate_move(action, node.board)
            if not changed:
                break

            spawned = self._spawn_tile(afterstate)
            board_key = self._board_key(spawned)
            action_children = node.children.setdefault(action, {})
            child = action_children.get(board_key)

            path.append((node, action))
            total_reward += gained
            depth += 1

            if child is None:
                child = Node(spawned)
                action_children[board_key] = child
                total_reward += self._rollout(spawned, self.config.rollout_depth - depth)
                node = child
                break

            node = child

        path.append((node, None))
        for visited_node, action in path:
            visited_node.visits += 1
            visited_node.value += total_reward
            if action is not None:
                stats = visited_node.action_stats.setdefault(action, ActionStats())
                stats.visits += 1
                stats.value += total_reward

        return total_reward

    def _select_action(self, node: Node, legal: list[Action]) -> Action:
        unvisited = [action for action in legal if node.action_stats.get(action, ActionStats()).visits == 0]
        if unvisited:
            return self._rng.choice(unvisited)

        log_parent = math.log(max(node.visits, 1))

        def ucb(action: Action) -> float:
            stats = node.action_stats[action]
            average = stats.value / stats.visits
            bonus = self.config.exploration * math.sqrt(log_parent / stats.visits)
            return average + bonus

        return max(legal, key=ucb)

    def _rollout(self, board: np.ndarray, remaining_depth: int) -> float:
        total = 0.0
        current = board.copy()
        for _ in range(max(remaining_depth, 0)):
            legal = self._legal_actions(current)
            if not legal:
                break
            action = self._rollout_policy(current, legal)
            afterstate, gained, changed = self._rules.simulate_move(action, current)
            if not changed:
                break
            current = self._spawn_tile(afterstate)
            total += gained
        return total + self._board_heuristic(current)

    def _rollout_policy(self, board: np.ndarray, legal: list[Action]) -> Action:
        best_action = legal[0]
        best_score = -float("inf")
        for action in legal:
            afterstate, gained, _ = self._rules.simulate_move(action, board)
            score = gained + 16.0 * self._empty_cells(afterstate) + self._corner_bonus(afterstate)
            score += 2.0 * self._monotonicity(afterstate)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _spawn_tile(self, board: np.ndarray) -> np.ndarray:
        spawned = board.copy()
        empty = list(zip(*np.where(spawned == 0)))
        if not empty:
            return spawned
        row, col = self._rng.choice(empty)
        spawned[row, col] = 2 if self._rng.random() < self.config.p_spawn_2 else 4
        return spawned

    def _legal_actions(self, board: np.ndarray) -> list[Action]:
        return self._rules.legal_actions(board)

    @staticmethod
    def _board_key(board: np.ndarray) -> BoardKey:
        return tuple(int(value) for value in board.flatten())

    @staticmethod
    def _average_value(stats: Optional[ActionStats]) -> float:
        if stats is None or stats.visits == 0:
            return -float("inf")
        return stats.value / stats.visits

    @staticmethod
    def _empty_cells(board: np.ndarray) -> int:
        return int(np.sum(board == 0))

    @staticmethod
    def _corner_bonus(board: np.ndarray) -> float:
        max_tile = int(board.max())
        corners = (board[0, 0], board[0, -1], board[-1, 0], board[-1, -1])
        return 32.0 if max_tile in corners else 0.0

    @staticmethod
    def _monotonicity(board: np.ndarray) -> float:
        score = 0.0
        for line in list(board) + list(board.T):
            nonzero = [int(value) for value in line if value]
            if len(nonzero) < 2:
                continue
            score += sum(1.0 for left, right in zip(nonzero, nonzero[1:]) if left >= right)
        return score

    def _board_heuristic(self, board: np.ndarray) -> float:
        return (
            24.0 * self._empty_cells(board)
            + self._corner_bonus(board)
            + 2.0 * self._monotonicity(board)
            + math.log2(max(int(board.max()), 2))
        )


_DEFAULT_AGENT = MCTS2048Agent()


def choose_action(game: Game2048) -> Action:
    """Default callable for ``python -m gui.pygame_demo --mode agent``."""
    return _DEFAULT_AGENT.choose_action(game)
