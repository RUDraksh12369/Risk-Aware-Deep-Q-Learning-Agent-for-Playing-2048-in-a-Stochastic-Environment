"""Load a trained DQN checkpoint and expose a GUI-compatible greedy policy."""

from __future__ import annotations

import os
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from game.environment import encode_board
from game.game import Action, Game2048
from inference.heuristics import HeuristicWeights, choose_tiebroken_action
from rl.agent import DQNAgent
from rl.risk_aware import RiskAwareDQNAgent

_agent: Optional[DQNAgent] = None
_agent_key: tuple[Path, Optional[str]] | None = None
ENHANCED_CHECKPOINT = Path("models/dqn_2048_enhanced.pt")
BASELINE_CHECKPOINT = Path("models/dqn_baseline_2048.pt")
RA_D3QN_CHECKPOINT = Path("models/ra_d3qn_2048.pt")
DEFAULT_RISK_COEFFICIENT = 0.20
_last_logged_inference: tuple[str, str] | None = None


def load_policy(checkpoint: str | Path | None = None, device: Optional[str] = None) -> DQNAgent:
    """Load a checkpoint once and return its DQN agent."""
    global _agent, _agent_key
    if checkpoint is None:
        if RA_D3QN_CHECKPOINT.is_file():
            path = RA_D3QN_CHECKPOINT
        else:
            path = ENHANCED_CHECKPOINT if ENHANCED_CHECKPOINT.is_file() else BASELINE_CHECKPOINT
    else:
        path = Path(checkpoint)
    key = (path.resolve(), device)
    if _agent is None or _agent_key != key:
        if not path.is_file():
            raise FileNotFoundError(
                f"DQN checkpoint not found: {path}. Train one with "
                f"`python -m training.train --checkpoint {BASELINE_CHECKPOINT.as_posix()}`."
            )
        _agent = DQNAgent.load(path, device=device)
        _agent_key = key
    return _agent


def choose_action_original(game: Game2048) -> Action:
    """Use the checkpoint's original deterministic, legal-masked policy."""
    agent = load_policy(os.environ.get("DQN_CHECKPOINT"))
    _log_inference_once("original", agent)
    return _choose_with_agent(agent, game)


def choose_action(game: Game2048) -> Action:
    """Use the validated RA-D3QN policy; other model types retain baseline play.

    Set ``DQN_INFERENCE_POLICY=original`` to select the preserved baseline.
    """
    agent = load_policy(os.environ.get("DQN_CHECKPOINT"))
    requested_policy = os.environ.get("DQN_INFERENCE_POLICY", "optimized").strip().lower()
    if requested_policy not in {"optimized", "original"}:
        raise ValueError("DQN_INFERENCE_POLICY must be 'optimized' or 'original'.")
    if requested_policy == "original" or not isinstance(agent, RiskAwareDQNAgent):
        _log_inference_once(requested_policy if isinstance(agent, RiskAwareDQNAgent) else "original", agent)
        return _choose_with_agent(agent, game)

    risk_coefficient = _float_setting("DQN_RISK_COEFFICIENT", DEFAULT_RISK_COEFFICIENT)
    tolerance = _float_setting("DQN_TIE_TOLERANCE", 0.01)
    weights = HeuristicWeights(
        monotonicity=_float_setting("DQN_HEURISTIC_MONOTONICITY", 1.0),
        smoothness=_float_setting("DQN_HEURISTIC_SMOOTHNESS", 1.0),
        empty_cells=_float_setting("DQN_HEURISTIC_EMPTY_CELLS", 1.0),
        highest_tile_corner=_float_setting("DQN_HEURISTIC_CORNER", 1.0),
    )
    _log_inference_once("optimized", agent, risk_coefficient, tolerance, weights)
    return choose_tiebroken_action(
        agent, game, tie_tolerance=tolerance, weights=weights,
        risk_coefficient=risk_coefficient,
    )


def _choose_with_agent(agent: DQNAgent, game: Game2048) -> Action:
    legal = game.legal_actions()
    if not legal:
        raise ValueError("DQN cannot choose an action from a terminal board.")
    mask = [action in legal for action in Action]
    return Action(agent.select_action(encode_board(game.board), mask, epsilon=0.0, game=game))


def _float_setting(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _log_inference_once(
    policy: str, agent: DQNAgent, risk_coefficient: float | None = None,
    tie_tolerance: float | None = None, weights: HeuristicWeights | None = None,
) -> None:
    global _last_logged_inference
    checkpoint = os.environ.get("DQN_CHECKPOINT") or str(
        RA_D3QN_CHECKPOINT if isinstance(agent, RiskAwareDQNAgent) else BASELINE_CHECKPOINT
    )
    config = (
        f"risk={risk_coefficient:.2f};tie={tie_tolerance:.3f};weights={asdict(weights)}"
        if policy == "optimized" and weights is not None else ""
    )
    key = (policy, f"{checkpoint}|{config}")
    if key == _last_logged_inference:
        return
    if policy == "optimized":
        print(
            f"[inference] policy=RA-D3QN optimized checkpoint={checkpoint} "
            f"risk_coefficient={risk_coefficient:.2f} tie_tolerance={tie_tolerance:.3f} "
            f"weights={asdict(weights)}",
            flush=True,
        )
    else:
        print(f"[inference] policy=original checkpoint={checkpoint}", flush=True)
    _last_logged_inference = key
