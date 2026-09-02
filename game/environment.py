"""
game/environment.py

RL-facing wrapper around the pure Game2048 engine (game.py).

Responsibilities that belong HERE (not in game.py):
- state encoding for the neural network (log2 encoding, Section 6)
- reward function (Section 13) — Model A (raw score) is implemented now;
  Model B (risk-shaped reward) is added later once rl/risk.py exists, via
  the `reward_fn` hook so we don't have to touch this file again.
- a Gym-like reset()/step() API that RL code (rl/agent.py) can depend on.
- `simulate_action(action)`: returns the encoded afterstate for each action
  WITHOUT mutating the environment — this is what Section 11's
  action-specific risk needs.

Keeping this separate from game.py means:
- game.py stays a dependency-free rules engine, reusable by Expectimax /
  heuristic baselines that don't care about RL-specific concerns.
- changing the reward/encoding later never risks breaking the game rules.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from game.game import Action, Game2048

# Max representable tile for log2 encoding headroom (2^16 = 65536, far
# beyond any realistic 4x4 game, so this never clips in practice).
LOG2_MAX_EXPONENT = 16


def encode_board(board: np.ndarray) -> np.ndarray:
    """Log2-encode a raw board (Section 6).

    0 -> 0, 2 -> 1, 4 -> 2, 8 -> 3, ..., 2048 -> 11, ...
    Returned as float32, flattened to shape (16,) for an MLP-DQN.
    (CNN-DQN, if/when implemented, should reshape this back to (4,4)
    rather than duplicating the encoding logic.)
    """
    with np.errstate(divide="ignore"):
        encoded = np.where(board > 0, np.log2(board), 0)
    encoded = np.clip(encoded, 0, LOG2_MAX_EXPONENT).astype(np.float32)
    return encoded.flatten()


class Env2048:
    """Gym-like RL environment around Game2048."""

    def __init__(
        self,
        size: int = 4,
        p_spawn_2: float = 0.9,
        seed: Optional[int] = None,
        illegal_move_penalty: float = 0.0,
        game_over_penalty: float = 0.0,
        reward_fn: Optional[Callable[["Env2048", int, bool], float]] = None,
    ):
        """
        Args:
            illegal_move_penalty: reward subtracted if the agent picks an
                action that doesn't change the board. Kept at 0 by default
                (Model A / Section 13 uses plain game score); the training
                loop can enable a small penalty to discourage wasted moves
                without changing the "core" reward semantics.
            game_over_penalty: reward subtracted when the game ends.
            reward_fn: optional override hook of signature
                (env, raw_score_gain, done) -> float. This is where
                Model B's risk-shaped reward (R' = R_game - lambda*Risk(s'))
                will plug in later without modifying this class.
        """
        self.game = Game2048(size=size, p_spawn_2=p_spawn_2, seed=seed)
        self.illegal_move_penalty = illegal_move_penalty
        self.game_over_penalty = game_over_penalty
        self.reward_fn = reward_fn

        self.action_space: List[Action] = list(Action)
        self.state_dim = size * size
        self.n_actions = len(self.action_space)

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        board = self.game.reset()
        return encode_board(board)

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict]:
        raw_board, raw_gain, done, info = self.game.step(action)

        reward = float(raw_gain)
        if not info["valid"]:
            reward -= self.illegal_move_penalty
        if done:
            reward -= self.game_over_penalty

        if self.reward_fn is not None:
            reward = self.reward_fn(self, raw_gain, done)

        info["raw_score_gain"] = raw_gain
        info["max_tile"] = self.game.max_tile()
        info["score"] = self.game.score

        return encode_board(raw_board), reward, done, info

    # ------------------------------------------------------------------ #
    # Action-specific simulation — required by rl/risk.py (Section 11)
    # ------------------------------------------------------------------ #
    def simulate_action(self, action: Action) -> Tuple[np.ndarray, np.ndarray, int, bool]:
        """Return the hypothetical afterstate of taking `action` from the
        CURRENT real state, without mutating anything and without spawning
        a random tile (the risk module reasons about the deterministic
        afterstate, matching Section 11's diagram).

        Returns:
            raw_afterstate: (4,4) int board
            encoded_afterstate: (16,) float32 board for feeding a network
            score_gain: int, score gained by this hypothetical move
            valid: bool, whether the move actually changes the board
        """
        raw_afterstate, score_gain, valid = self.game.simulate_move(action)
        return raw_afterstate, encode_board(raw_afterstate), score_gain, valid

    def legal_action_mask(self) -> np.ndarray:
        """Boolean mask over `self.action_space`, True where the action is
        legal (changes the board) from the current state. Useful for
        masking illegal actions during both epsilon-greedy exploration and
        greedy exploitation.
        """
        legal = set(self.game.legal_actions())
        return np.array([a in legal for a in self.action_space], dtype=bool)

    # ------------------------------------------------------------------ #
    def render(self) -> None:
        print(self.game)

    @property
    def board(self) -> np.ndarray:
        return self.game.board

    @property
    def done(self) -> bool:
        return self.game.done

    @property
    def score(self) -> int:
        return self.game.score


if __name__ == "__main__":
    import random as _random

    env = Env2048(seed=7)
    state = env.reset()
    print("Encoded initial state:", state)
    env.render()

    total_reward = 0.0
    steps = 0
    while not env.done and steps < 200:
        mask = env.legal_action_mask()
        legal_actions = [a for a, ok in zip(env.action_space, mask) if ok]
        if not legal_actions:
            break
        action = _random.choice(legal_actions)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"\nEpisode finished after {steps} steps.")
    print(f"Total (raw) reward: {total_reward}")
    print(f"Final score: {env.score}, max tile: {env.game.max_tile()}")
    env.render()
