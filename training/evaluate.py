"""Greedy, repeatable evaluation for saved 2048 DQN checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from game.environment import Env2048
from rl.agent import DQNAgent
from rl.baselines import GreedyAgent, RandomAgent
from rl.risk import board_risk


def evaluate(
    agent: DQNAgent | RandomAgent | GreedyAgent, episodes: int = 100, seed: int = 1_000,
    *, max_steps_per_episode: int | None = None, high_risk_threshold: float = 0.5,
    episode_records: list[dict[str, float | int | bool]] | None = None,
) -> dict[str, float]:
    """Play greedy episodes and report performance and pre-action board risk.

    Risk metrics pool nonterminal decision states across episodes, using the
    same default RiskConfig for every agent (independent of policy weights).
    Low mobility means at most one legal move; high risk means score >= the
    threshold. Game-over excludes time-limit truncation; without a time limit,
    complete 2048 episodes naturally have a game-over rate of one.
    When supplied, episode_records receives one record per episode without
    changing the existing aggregate return value or policy/environment RNG.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    if max_steps_per_episode is not None and max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be positive or None.")
    if not 0.0 <= high_risk_threshold <= 1.0:
        raise ValueError("high_risk_threshold must be in [0, 1].")
    scores: list[float] = []
    max_tiles: list[float] = []
    episode_lengths: list[int] = []
    game_overs: list[bool] = []
    legal_move_counts: list[int] = []
    empty_cell_counts: list[int] = []
    risks: list[float] = []
    for offset in range(episodes):
        start_index = len(risks)
        env = Env2048(seed=seed + offset)
        state = env.reset()
        steps = 0
        while not env.done:
            if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                break
            mask = env.legal_action_mask()
            if not mask.any():
                break
            board_metrics = board_risk(env.game)
            legal_move_counts.append(board_metrics.legal_moves)
            empty_cell_counts.append(board_metrics.empty_cells)
            risks.append(board_metrics.score)
            action = agent.select_action(state, mask, epsilon=0.0, game=env.game)
            state, _, done, _ = env.step(action)
            steps += 1
            if done:
                break
        scores.append(float(env.score))
        max_tiles.append(float(env.game.max_tile()))
        episode_lengths.append(steps)
        game_overs.append(env.done or not env.legal_action_mask().any())
        if episode_records is not None:
            episode_risks = np.asarray(risks[start_index:])
            episode_moves = np.asarray(legal_move_counts[start_index:])
            episode_empty = np.asarray(empty_cell_counts[start_index:])
            episode_records.append({
                "episode": offset + 1, "seed": seed + offset,
                "score": scores[-1], "max_tile": max_tiles[-1], "steps": steps,
                "terminated": bool(game_overs[-1]), "truncated": not bool(game_overs[-1]),
                "decision_states": len(episode_risks),
                "mean_legal_moves": float(episode_moves.mean()) if steps else 0.0,
                "mean_empty_cells": float(episode_empty.mean()) if steps else 0.0,
                "mean_board_risk": float(episode_risks.mean()) if steps else 0.0,
                "low_mobility_count": int(np.count_nonzero(episode_moves <= 1)),
                "high_risk_count": int(np.count_nonzero(episode_risks >= high_risk_threshold)),
                "high_risk_threshold": high_risk_threshold,
            })

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
        "tile_2048_rate": float(np.mean(tile_array >= 2048)),
        "max_tile": float(tile_array.max()),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "game_over_rate": float(np.mean(game_overs)),
        "truncation_rate": float(1.0 - np.mean(game_overs)),
        "mean_legal_moves": float(np.mean(legal_move_counts)) if legal_move_counts else 0.0,
        "mean_empty_cells": float(np.mean(empty_cell_counts)) if empty_cell_counts else 0.0,
        "mean_board_risk": float(np.mean(risks)) if risks else 0.0,
        "low_mobility_rate": float(np.mean(np.asarray(legal_move_counts) <= 1)) if risks else 0.0,
        "high_risk_rate": float(np.mean(np.asarray(risks) >= high_risk_threshold)) if risks else 0.0,
        "high_risk_threshold": float(high_risk_threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/dqn_2048_enhanced.pt"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--high-risk-threshold", type=float, default=0.5)
    parser.add_argument("--episodes-csv", type=Path, default=None, help="Export individual episodes to a new CSV file.")
    args = parser.parse_args()
    if args.episodes_csv is not None and args.episodes_csv.exists():
        parser.error("episodes-csv output already exists; choose a new file")
    episode_records = [] if args.episodes_csv is not None else None
    metrics = evaluate(
        DQNAgent.load(args.checkpoint, device=args.device), args.episodes, args.seed,
        max_steps_per_episode=args.max_steps_per_episode, high_risk_threshold=args.high_risk_threshold,
        episode_records=episode_records,
    )
    if args.episodes_csv is not None:
        args.episodes_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.episodes_csv.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(episode_records[0]))
            writer.writeheader()
            writer.writerows(episode_records)
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
