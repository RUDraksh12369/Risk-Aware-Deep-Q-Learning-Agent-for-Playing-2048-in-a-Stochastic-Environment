"""Compare deterministic checkpoint inference policies on shared game seeds."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from game.environment import Env2048
from game.game import Action
from inference.heuristics import HeuristicWeights, choose_tiebroken_action
from rl.agent import DQNAgent


def _masked_greedy(agent: DQNAgent, env: Env2048, state: np.ndarray) -> Action:
    mask = env.legal_action_mask()
    values = agent._policy_q_values(state, env.game).copy()
    values[~mask] = -np.inf
    return Action(int(np.argmax(values)))


def evaluate_policies(
    agent: DQNAgent,
    *,
    episodes: int,
    seed: int,
    tie_tolerance: float,
    weights: HeuristicWeights,
) -> tuple[list[dict[str, int | float | str]], list[dict[str, int | float | str]]]:
    """Return aggregate and per-game rows; each policy sees the same seeds."""
    policies = {
        "original_inference": lambda env, state: Action(int(agent.select_action(
            state, env.legal_action_mask(), epsilon=0.0, game=env.game))),
        "legal_masked_greedy": lambda env, state: _masked_greedy(agent, env, state),
        "heuristic_tiebreak": lambda env, state: choose_tiebroken_action(
            agent, env.game, tie_tolerance=tie_tolerance, weights=weights),
    }
    per_game: list[dict[str, int | float | str]] = []
    summaries: list[dict[str, int | float | str]] = []
    for policy_name, choose in policies.items():
        scores, max_tiles, moves = [], [], []
        for index in range(episodes):
            game_seed = seed + index
            env = Env2048(seed=game_seed)
            state = env.reset()
            move_count = 0
            while not env.done:
                mask = env.legal_action_mask()
                if not mask.any():
                    break
                action = choose(env, state)
                state, _, _, _ = env.step(action)
                move_count += 1

            score, max_tile = float(env.score), int(env.game.max_tile())
            scores.append(score)
            max_tiles.append(max_tile)
            moves.append(move_count)
            per_game.append({
                "policy": policy_name, "episode": index + 1, "seed": game_seed,
                "final_score": score, "max_tile": max_tile, "moves": move_count,
            })

        row: dict[str, int | float | str] = {
            "policy": policy_name,
            "episodes": episodes,
            "average_final_score": float(np.mean(scores)),
            "maximum_tile_achieved": int(max(max_tiles)),
            "average_maximum_tile": float(np.mean(max_tiles)),
            "average_moves": float(np.mean(moves)),
        }
        for threshold in (512, 1024, 2048, 4096, 8192):
            row[f"games_reaching_{threshold}_or_higher"] = int(sum(tile >= threshold for tile in max_tiles))
        summaries.append(row)
        print(f"{policy_name}: avg score {row['average_final_score']:.1f}, "
              f"avg max tile {row['average_maximum_tile']:.1f}, "
              f"max tile {row['maximum_tile_achieved']}", flush=True)
    return summaries, per_game


def _write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tie-tolerance", type=float, default=0.01,
                        help="Relative Q-value window in which heuristics may break close decisions.")
    parser.add_argument("--monotonicity-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=1.0)
    parser.add_argument("--empty-cells-weight", type=float, default=1.0)
    parser.add_argument("--corner-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or not np.isfinite(args.tie_tolerance) or args.tie_tolerance < 0:
        parser.error("episodes must be positive, seed nonnegative, and tie tolerance finite/nonnegative")
    weight_values = (args.monotonicity_weight, args.smoothness_weight,
                     args.empty_cells_weight, args.corner_weight)
    if not all(np.isfinite(value) and value >= 0 for value in weight_values):
        parser.error("heuristic weights must be finite and nonnegative")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    output = args.output_dir
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"output directory already exists: {output}")

    weights = HeuristicWeights(*weight_values)
    agent = DQNAgent.load(args.checkpoint, device=args.device)
    summaries, per_game = evaluate_policies(
        agent, episodes=args.episodes, seed=args.seed,
        tie_tolerance=args.tie_tolerance, weights=weights,
    )
    _write_csv(output / "summary.csv", summaries)
    _write_csv(output / "games.csv", per_game)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "episodes_per_policy": args.episodes,
        "shared_seed_range": [args.seed, args.seed + args.episodes - 1],
        "tie_tolerance_relative": args.tie_tolerance,
        "heuristic_weights": asdict(weights),
        "notes": [
            "No training or checkpoint writes are performed.",
            "The original inference implementation already uses epsilon=0 and legal-action masking.",
            "Original and explicitly masked greedy rows should therefore match exactly.",
            "Heuristics are consulted only among legal actions inside the configured Q-value window.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote comparison to {output}")


if __name__ == "__main__":
    main()
