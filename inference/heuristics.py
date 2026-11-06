"""Optional inference-only tie-breaking for already-trained 2048 policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from game.environment import encode_board
from game.game import Action, Game2048


@dataclass(frozen=True)
class HeuristicWeights:
    """Weights used only when legal action Q-values are close."""

    monotonicity: float = 1.0
    smoothness: float = 1.0
    empty_cells: float = 1.0
    highest_tile_corner: float = 1.0


def board_heuristic_score(board: np.ndarray, weights: HeuristicWeights) -> float:
    """Return a normalized board preference score; larger is better."""
    values = np.asarray(board, dtype=np.float64)
    logs = np.zeros_like(values)
    occupied = values > 0
    logs[occupied] = np.log2(values[occupied])

    # For each row and column, prefer either consistently increasing or
    # consistently decreasing tile values, allowing either orientation.
    monotonicity = 0.0
    for line in (*logs, *logs.T):
        differences = np.diff(line)
        monotonicity += max(-float(np.maximum(differences, 0).sum()),
                            -float(np.maximum(-differences, 0).sum()))
    monotonicity /= max(1.0, values.size * 16.0)

    neighbor_differences = []
    for a, b in ((values[:, :-1], values[:, 1:]), (values[:-1, :], values[1:, :])):
        both_occupied = (a > 0) & (b > 0)
        if both_occupied.any():
            neighbor_differences.extend(np.abs(np.log2(a[both_occupied]) - np.log2(b[both_occupied])))
    smoothness = -float(np.sum(neighbor_differences)) / max(1.0, values.size * 16.0)

    empty_cells = float(np.count_nonzero(~occupied)) / values.size
    corners = values[[0, 0, -1, -1], [0, -1, 0, -1]]
    highest_in_corner = float(values.max() > 0 and np.any(corners == values.max()))

    return (
        weights.monotonicity * monotonicity
        + weights.smoothness * smoothness
        + weights.empty_cells * empty_cells
        + weights.highest_tile_corner * highest_in_corner
    )


def choose_tiebroken_action(
    agent,
    game: Game2048,
    *,
    tie_tolerance: float = 0.01,
    weights: HeuristicWeights | None = None,
    risk_coefficient: float | None = None,
) -> Action:
    """Choose a greedy legal action, consulting board features only near ties.

    ``tie_tolerance`` is a relative Q-value gap. Actions within that fraction
    of ``max(1, abs(best_q))`` from the best legal Q-value are eligible.
    By default, policy-specific Q adjustments (for example RA-D3QN) are
    preserved through the agent's existing ``_policy_q_values`` hook. For
    RA-D3QN, ``risk_coefficient`` can override its saved coefficient in memory
    for this decision without modifying the checkpoint or agent configuration.
    """
    weight_values = (
        weights.monotonicity, weights.smoothness,
        weights.empty_cells, weights.highest_tile_corner,
    ) if weights is not None else ()
    if any(not np.isfinite(value) or value < 0 for value in weight_values):
        raise ValueError("heuristic weights must be finite and nonnegative.")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative.")
    legal = game.legal_actions()
    if not legal:
        raise ValueError("Cannot select an action from a terminal board.")
    state = encode_board(game.board)
    if risk_coefficient is not None:
        risk_config = getattr(agent, "risk_config", None)
        if risk_config is None:
            raise ValueError("risk_coefficient override requires an RA-D3QN agent.")
        from rl.risk import risk_adjusted_q_values

        values = risk_adjusted_q_values(
            game, agent.q_values(state), risk_coefficient, risk_config.weights
        )
    else:
        values = np.asarray(agent._policy_q_values(state, game), dtype=np.float64).copy()
    if values.shape != (len(Action),):
        raise ValueError("Policy returned invalid Q-values.")
    legal_ids = np.asarray([int(action) for action in legal], dtype=np.int64)
    if not np.isfinite(values[legal_ids]).all():
        raise ValueError("Policy returned invalid legal-action Q-values.")
    best_q = float(values[legal_ids].max())
    candidates = legal_ids[values[legal_ids] >= best_q - tie_tolerance * max(1.0, abs(best_q))]
    if candidates.size == 1 or tie_tolerance == 0:
        return Action(int(candidates[0]))

    weights = weights or HeuristicWeights()
    scored = []
    for action_id in candidates:
        afterstate, _, _ = game.simulate_move(Action(int(action_id)))
        scored.append(board_heuristic_score(afterstate, weights))
    return Action(int(candidates[int(np.argmax(scored))]))
