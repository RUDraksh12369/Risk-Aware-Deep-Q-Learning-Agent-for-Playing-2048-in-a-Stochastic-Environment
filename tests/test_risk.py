"""Deterministic checks of the proposal's action-risk evaluation mechanism."""

import numpy as np
import pytest

from game.game import Action, Game2048
from rl.risk import RiskConfig, action_risks, board_risk, risk_adjusted_q_values


CHECKERBOARD = np.array([
    [2, 4, 2, 4],
    [4, 2, 4, 2],
    [2, 4, 2, 4],
    [4, 2, 4, 2],
], dtype=np.int64)


def make_game(board, p_spawn_2=0.9):
    game = Game2048(seed=9, p_spawn_2=p_spawn_2)
    game.board = np.asarray(board, dtype=np.int64).copy()
    return game


def test_sparse_board_has_low_risk_and_terminal_board_has_maximum_risk():
    board = np.zeros((4, 4), dtype=np.int64)
    # Even an empty setup board has legal moves after a tile is spawned.
    assert board_risk(make_game(board)).spawn_game_over_probability == 0.0
    board[1, 1] = 2
    sparse = board_risk(make_game(board))
    assert sparse.legal_moves == 4
    assert sparse.empty_cells == 15
    assert sparse.mobility_risk == 0.0
    assert sparse.crowding_risk == 1.0 / 16
    assert sparse.spawn_game_over_probability == 0.0
    assert sparse.score == pytest.approx(1.0 / 48)
    terminal = board_risk(make_game(CHECKERBOARD))
    assert terminal.legal_moves == 0
    assert terminal.empty_cells == 0
    assert terminal.spawn_game_over_probability == 1.0
    assert terminal.score == 1.0


@pytest.mark.parametrize("p_spawn_2", [0.0, 0.3, 0.9, 1.0])
@pytest.mark.parametrize("multiplier", [1, 2, 4])
def test_single_empty_cell_has_exact_spawn_game_over_probability(p_spawn_2, multiplier):
    # The last cell's neighbors are 4*multiplier. Only a matching spawn
    # saves this board; with multiplier>1 neither 2 nor 4 can save it.
    board = CHECKERBOARD * multiplier
    board[-1, -1] = 0
    expected = p_spawn_2 if multiplier == 1 else 1.0
    result = board_risk(make_game(board, p_spawn_2))
    assert result.spawn_game_over_probability == pytest.approx(expected)


def test_spawn_of_two_can_be_the_only_safe_outcome():
    board = CHECKERBOARD.copy()
    board[0, -1] = 0  # Both neighbors are 2.
    assert board_risk(make_game(board, 0.7)).spawn_game_over_probability == pytest.approx(0.3)


def test_two_empty_cells_cannot_end_the_game_after_one_spawn():
    board = CHECKERBOARD.copy()
    board[-1, -2:] = 0
    game = make_game(board)
    assert board_risk(game).spawn_game_over_probability == 0.0
    for row, col in np.argwhere(board == 0):
        for tile in (2, 4):
            candidate = board.copy()
            candidate[row, col] = tile
            assert game.legal_actions(candidate)


def test_full_board_with_a_merge_is_not_immediately_terminal():
    board = CHECKERBOARD.copy()
    board[0, 0] = board[0, 1]
    result = board_risk(make_game(board))
    assert result.empty_cells == 0
    assert result.legal_moves > 0
    assert result.spawn_game_over_probability == 0.0
    assert result.score < 1.0


def test_weights_can_isolate_each_component_and_are_scale_invariant():
    board = CHECKERBOARD.copy()
    board[-1, -1] = 0
    game = make_game(board)
    result = board_risk(game)
    assert board_risk(game, config=RiskConfig(1, 0, 0)).score == result.mobility_risk
    assert board_risk(game, config=RiskConfig(0, 1, 0)).score == result.crowding_risk
    assert board_risk(game, config=RiskConfig(0, 0, 1)).score == result.spawn_game_over_probability
    assert board_risk(game, config=RiskConfig(2, 2, 2)).score == pytest.approx(result.score)


@pytest.mark.parametrize("weights", [(-1, 1, 1), (0, 0, 0), (float("nan"), 1, 1),
                                    (1, float("inf"), 1)])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        RiskConfig(*weights)


def test_action_risk_prefers_more_space_and_masks_illegal_actions():
    game = make_game([
        [2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [2, 2, 4, 0],
    ])
    risks = action_risks(game)
    assert np.isposinf(risks[Action.UP])
    assert risks[Action.DOWN] == pytest.approx((0.25 + 15 / 16) / 3)
    assert risks[Action.LEFT] == pytest.approx((0.25 + 14 / 16) / 3)
    assert risks[Action.RIGHT] == risks[Action.LEFT]
    values = np.array([100.0, 1.01, 1.0, 0.5])
    before = values.copy()
    baseline = risk_adjusted_q_values(game, values, 0.0)
    assert np.isneginf(baseline[Action.UP])
    np.testing.assert_array_equal(baseline[1:], values[1:])
    assert np.argmax(baseline) == Action.DOWN
    adjusted = risk_adjusted_q_values(game, values, 1.0)
    assert np.argmax(adjusted) == Action.LEFT
    np.testing.assert_array_equal(values, before)


def test_risk_evaluation_preserves_board_score_moves_and_future_random_spawns():
    game = make_game(CHECKERBOARD)
    game.board[-1, -1] = 0
    game.score = 123
    game.moves_made = 17
    original = game.clone()
    for _ in range(3):
        board_risk(game)
        action_risks(game)
        risk_adjusted_q_values(game, np.zeros(4), 0.5)
    np.testing.assert_array_equal(game.board, original.board)
    assert (game.score, game.moves_made, game.done) == (123, 17, original.done)
    assert game._rng.getstate() == original._rng.getstate()
    action = game.legal_actions()[0]
    actual = game.step(action)
    expected = original.step(action)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]


def test_no_legal_actions_are_reported_without_selecting_an_invalid_move():
    game = make_game(CHECKERBOARD)
    assert np.isposinf(action_risks(game)).all()
    with pytest.raises(ValueError, match="no legal moves"):
        risk_adjusted_q_values(game, np.zeros(4), 0.0)


@pytest.mark.parametrize("coefficient", [-1.0, float("nan"), float("inf")])
def test_invalid_risk_coefficient_is_rejected(coefficient):
    with pytest.raises(ValueError, match="risk_coefficient"):
        risk_adjusted_q_values(Game2048(seed=0), np.zeros(4), coefficient)


@pytest.mark.parametrize("values", [np.zeros(3), np.zeros((1, 4)), [0, 1, 2, np.nan]])
def test_invalid_q_values_are_rejected(values):
    with pytest.raises(ValueError, match="q_values"):
        risk_adjusted_q_values(Game2048(seed=0), values, 0.1)


@pytest.mark.parametrize("board", [np.zeros((3, 3), dtype=int), np.zeros((4, 4)),
                                 np.full((4, 4), 3), np.full((4, 4), -2)])
def test_invalid_raw_boards_are_rejected(board):
    with pytest.raises(ValueError):
        board_risk(Game2048(seed=0), board)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_invalid_spawn_probability_is_rejected(probability):
    game = Game2048(seed=0)
    game.p_spawn_2 = probability
    with pytest.raises(ValueError, match="p_spawn_2"):
        board_risk(game)
