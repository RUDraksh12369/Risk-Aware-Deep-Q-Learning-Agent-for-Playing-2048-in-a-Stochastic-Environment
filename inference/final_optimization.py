"""Inference-only RA-D3QN risk/heuristic comparison; never writes checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np

from game.environment import Env2048
from game.game import Action
from inference.heuristics import HeuristicWeights, choose_tiebroken_action
from rl.agent import DQNAgent
from rl.risk_aware import RiskAwareDQNAgent


def _run_config(
    agent: RiskAwareDQNAgent,
    *,
    name: str,
    risk_coefficient: float,
    use_heuristic: bool,
    episodes: int,
    seed: int,
    tie_tolerance: float,
    weights: HeuristicWeights,
    phase: str,
) -> tuple[dict, list[dict]]:
    original_risk = agent.risk_config
    agent.risk_config = replace(original_risk, risk_coefficient=risk_coefficient)
    scores: list[int] = []
    max_tiles: list[int] = []
    moves: list[int] = []
    games: list[dict] = []
    try:
        for episode in range(episodes):
            game_seed = seed + episode
            env = Env2048(seed=game_seed)
            state = env.reset()
            episode_moves = 0
            while not env.done:
                legal_mask = env.legal_action_mask()
                if not legal_mask.any():
                    break
                if use_heuristic:
                    action = choose_tiebroken_action(
                        agent, env.game, tie_tolerance=tie_tolerance, weights=weights
                    )
                else:
                    action = Action(agent.select_action(
                        state, legal_mask, epsilon=0.0, game=env.game
                    ))
                state, _, _, _ = env.step(action)
                episode_moves += 1

            score = int(env.score)
            max_tile = int(env.game.max_tile())
            scores.append(score)
            max_tiles.append(max_tile)
            moves.append(episode_moves)
            games.append({
                "phase": phase,
                "configuration": name,
                "episode": episode + 1,
                "seed": game_seed,
                "final_score": score,
                "max_tile": max_tile,
                "moves": episode_moves,
            })
            if (episode + 1) % 25 == 0:
                print(f"{phase} {name}: {episode + 1}/{episodes} games", flush=True)
    finally:
        agent.risk_config = original_risk

    histogram = {str(tile): int(sum(value == tile for value in max_tiles))
                 for tile in sorted(set(max_tiles))}
    summary = {
        "phase": phase,
        "configuration": name,
        "risk_coefficient": risk_coefficient,
        "heuristic_tiebreak": use_heuristic,
        "episodes": episodes,
        "seed_start": seed,
        "seed_end": seed + episodes - 1,
        "average_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "maximum_score": int(max(scores)),
        "score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "average_max_tile": float(np.mean(max_tiles)),
        "median_max_tile": float(np.median(max_tiles)),
        "maximum_tile_achieved": int(max(max_tiles)),
        "highest_tile_distribution": histogram,
        "games_reaching_512": int(sum(tile >= 512 for tile in max_tiles)),
        "games_reaching_1024": int(sum(tile >= 1024 for tile in max_tiles)),
        "games_reaching_2048": int(sum(tile >= 2048 for tile in max_tiles)),
        "average_moves": float(np.mean(moves)),
    }
    print(
        f"{phase} {name}: avg score={summary['average_score']:.1f}, "
        f"median={summary['median_score']:.1f}, avg max tile={summary['average_max_tile']:.1f}, "
        f"512+/1024+/2048+={summary['games_reaching_512']}/"
        f"{summary['games_reaching_1024']}/{summary['games_reaching_2048']}",
        flush=True,
    )
    return summary, games


def _selection_scores(rows: list[dict]) -> dict[str, float]:
    """Equal-weight mean percentile ranks across consistency and outcome metrics."""
    metrics = (
        ("average_score", True),
        ("median_score", True),
        ("average_max_tile", True),
        ("median_max_tile", True),
        ("games_reaching_512", True),
        ("games_reaching_1024", True),
        ("games_reaching_2048", True),
        ("score_std", False),
    )
    result = {row["configuration"]: [] for row in rows}
    for metric, higher_is_better in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        ordered = values if higher_is_better else -values
        ranks = np.zeros(len(rows), dtype=float)
        distinct = np.unique(ordered)
        for value in distinct:
            indices = np.flatnonzero(ordered == value)
            average_rank = (float(indices.min() + indices.max()) / 2.0) / max(1, len(rows) - 1)
            ranks[indices] = average_rank
        for row, rank in zip(rows, ranks):
            result[row["configuration"]].append(float(rank))
    return {name: float(np.mean(ranks)) for name, ranks in result.items()}


def _write_outputs(output: Path, summaries: list[dict], games: list[dict], manifest: dict) -> None:
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (output / "games.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(games[0]))
        writer.writeheader()
        writer.writerows(games)
    # A flat CSV complements the JSON's exact-tile histogram.
    flat = [{key: value for key, value in row.items() if key != "highest_tile_distribution"}
            | {f"max_tile_{tile}": count for tile, count in row["highest_tile_distribution"].items()}
            for row in summaries]
    columns = list(dict.fromkeys(key for row in flat for key in row))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--holdout-seed", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--tie-tolerance", type=float, default=0.01)
    parser.add_argument("--monotonicity-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=1.0)
    parser.add_argument("--empty-cells-weight", type=float, default=1.0)
    parser.add_argument("--corner-weight", type=float, default=1.0)
    args = parser.parse_args()
    weights = HeuristicWeights(
        args.monotonicity_weight, args.smoothness_weight,
        args.empty_cells_weight, args.corner_weight,
    )
    if args.episodes < 1 or args.seed < 0 or args.holdout_seed < 0:
        parser.error("episodes must be positive and seed values nonnegative")
    if (args.seed < args.holdout_seed + args.episodes
            and args.holdout_seed < args.seed + args.episodes):
        parser.error("primary and holdout seed ranges must not overlap")
    if not np.isfinite(args.tie_tolerance) or args.tie_tolerance < 0:
        parser.error("tie tolerance must be finite and nonnegative")
    if any(not np.isfinite(value) or value < 0 for value in asdict(weights).values()):
        parser.error("heuristic weights must be finite and nonnegative")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"output directory already exists: {args.output_dir}")

    agent = DQNAgent.load(args.checkpoint, device=args.device)
    if not isinstance(agent, RiskAwareDQNAgent):
        parser.error("checkpoint must be an RA-D3QN model")
    checkpoint_risk = agent.risk_config.risk_coefficient
    if checkpoint_risk != 0.1:
        print(f"Checkpoint stores risk coefficient {checkpoint_risk}, not the expected 0.1.", flush=True)

    summaries: list[dict] = []
    games: list[dict] = []
    manifest = {
        "status": "running",
        "checkpoint": str(args.checkpoint),
        "device": str(agent.device),
        "checkpoint_risk_coefficient": checkpoint_risk,
        "risk_sweep": [0.0, 0.05, 0.10, 0.15, 0.20],
        "primary_seed_range": [args.seed, args.seed + args.episodes - 1],
        "holdout_seed_range": [args.holdout_seed, args.holdout_seed + args.episodes - 1],
        "episodes_per_configuration_per_phase": args.episodes,
        "tie_tolerance_relative": args.tie_tolerance,
        "heuristic_weights": asdict(weights),
        "model_weights_modified": False,
        "training_invoked": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        risks = (0.0, 0.05, 0.10, 0.15, 0.20)
        risk_rows: list[dict] = []
        for coefficient in risks:
            config_name = "original_ra_d3qn" if coefficient == checkpoint_risk else f"risk_{coefficient:.2f}"
            row, per_game = _run_config(
                agent, name=config_name, risk_coefficient=coefficient, use_heuristic=False,
                episodes=args.episodes, seed=args.seed, tie_tolerance=args.tie_tolerance,
                weights=weights, phase="primary",
            )
            summaries.append(row)
            risk_rows.append(row)
            games.extend(per_game)
            if coefficient == checkpoint_risk:
                original_name = config_name

        heuristic_row, per_game = _run_config(
            agent, name="heuristic_at_original_risk", risk_coefficient=checkpoint_risk,
            use_heuristic=True, episodes=args.episodes, seed=args.seed,
            tie_tolerance=args.tie_tolerance, weights=weights, phase="primary",
        )
        summaries.append(heuristic_row)
        games.extend(per_game)

        risk_scores = _selection_scores(risk_rows)
        best_risk_row = max(risk_rows, key=lambda row: risk_scores[row["configuration"]])
        best_risk = float(best_risk_row["risk_coefficient"])
        combo_name = f"risk_{best_risk:.2f}_plus_heuristic"
        combo_row, per_game = _run_config(
            agent, name=combo_name, risk_coefficient=best_risk, use_heuristic=True,
            episodes=args.episodes, seed=args.seed, tie_tolerance=args.tie_tolerance,
            weights=weights, phase="primary",
        )
        summaries.append(combo_row)
        games.extend(per_game)

        candidates = [*summaries]
        candidate_scores = _selection_scores(candidates)
        selected = max(candidates, key=lambda row: candidate_scores[row["configuration"]])
        selected_name = str(selected["configuration"])
        selected_risk = float(selected["risk_coefficient"])
        selected_heuristic = bool(selected["heuristic_tiebreak"])
        holdout_original, per_game = _run_config(
            agent, name=original_name, risk_coefficient=checkpoint_risk, use_heuristic=False,
            episodes=args.episodes, seed=args.holdout_seed, tie_tolerance=args.tie_tolerance,
            weights=weights, phase="holdout",
        )
        summaries.append(holdout_original)
        games.extend(per_game)
        holdout_selected, per_game = _run_config(
            agent, name=selected_name, risk_coefficient=selected_risk,
            use_heuristic=selected_heuristic, episodes=args.episodes,
            seed=args.holdout_seed, tie_tolerance=args.tie_tolerance,
            weights=weights, phase="holdout",
        )
        summaries.append(holdout_selected)
        games.extend(per_game)

        manifest.update({
            "status": "completed",
            "risk_sweep_selection_scores": risk_scores,
            "best_risk_coefficient_before_heuristic": best_risk,
            "primary_selection_scores": candidate_scores,
            "selected_configuration": selected_name,
            "selected_risk_coefficient": selected_risk,
            "selected_heuristic_tiebreak": selected_heuristic,
            "holdout_original": original_name,
        })
    except Exception as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        _write_outputs(args.output_dir, summaries, games, manifest) if games else None
        raise

    _write_outputs(args.output_dir, summaries, games, manifest)
    print(f"Selected primary result: {selected_name}; validating on held-out seeds.", flush=True)
    print(f"Wrote comparison to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
