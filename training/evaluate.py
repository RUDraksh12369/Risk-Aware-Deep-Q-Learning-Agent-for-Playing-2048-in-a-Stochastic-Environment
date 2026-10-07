"""Greedy, repeatable evaluation for saved 2048 DQN checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from game.environment import Env2048
from rl.agent import DQNAgent


def evaluate(agent: DQNAgent, episodes: int = 100, seed: int = 1_000) -> dict[str, float]:
    """Play greedy episodes and return score/tile metrics in game-score units."""
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    scores: list[float] = []
    max_tiles: list[float] = []
    for offset in range(episodes):
        env = Env2048(seed=seed + offset)
        state = env.reset()
        while not env.done:
            mask = env.legal_action_mask()
            if not mask.any():
                break
            action = agent.select_action(state, mask, epsilon=0.0)
            state, _, done, _ = env.step(action)
            if done:
                break
        scores.append(float(env.score))
        max_tiles.append(float(env.game.max_tile()))

    score_array = np.asarray(scores)
    tile_array = np.asarray(max_tiles)
    return {
        "episodes": float(episodes),
        "mean_score": float(score_array.mean()),
        "median_score": float(np.median(score_array)),
        "best_score": float(score_array.max()),
        "mean_max_tile": float(tile_array.mean()),
        "tile_128_rate": float(np.mean(tile_array >= 128)),
        "tile_256_rate": float(np.mean(tile_array >= 256)),
        "tile_512_rate": float(np.mean(tile_array >= 512)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/dqn_2048_enhanced.pt"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1_000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    metrics = evaluate(DQNAgent.load(args.checkpoint, device=args.device), args.episodes, args.seed)
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
