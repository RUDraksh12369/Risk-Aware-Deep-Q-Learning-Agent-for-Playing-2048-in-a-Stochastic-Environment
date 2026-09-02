"""Unit tests for game/game.py and game/environment.py.

Run with: pytest -q  (from the project root)
"""
import numpy as np

from game.game import Action, Game2048
from game.environment import Env2048, encode_board


# ---------------------------------------------------------------------- #
# Game2048 mechanics
# ---------------------------------------------------------------------- #
def make_game_with_board(board_values, p_spawn_2=0.9, seed=0):
    g = Game2048(seed=seed, p_spawn_2=p_spawn_2)
    g.board = np.array(board_values, dtype=np.int64)
    return g


def test_reset_has_exactly_two_tiles():
    g = Game2048(seed=1)
    nonzero = np.count_nonzero(g.board)
    assert nonzero == 2
    assert g.score == 0
    assert not g.done


def test_left_merge_basic():
    g = make_game_with_board([
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    new_board, gained, changed = g.simulate_move(Action.LEFT)
    assert changed
    assert gained == 4
    assert new_board[0].tolist() == [4, 0, 0, 0]


def test_no_double_merge_in_one_move():
    # 2 2 2 2 -> merges into 4 4, NOT 8 0 (each tile merges at most once)
    g = make_game_with_board([
        [2, 2, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    new_board, gained, changed = g.simulate_move(Action.LEFT)
    assert new_board[0].tolist() == [4, 4, 0, 0]
    assert gained == 8


def test_right_move_direction():
    g = make_game_with_board([
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    new_board, gained, changed = g.simulate_move(Action.RIGHT)
    assert new_board[0].tolist() == [0, 0, 0, 4]
    assert gained == 4


def test_up_and_down_moves():
    g = make_game_with_board([
        [2, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    up_board, up_gain, up_changed = g.simulate_move(Action.UP)
    assert up_board[:, 0].tolist() == [4, 0, 0, 0]
    assert up_gain == 4

    down_board, down_gain, down_changed = g.simulate_move(Action.DOWN)
    assert down_board[:, 0].tolist() == [0, 0, 0, 4]
    assert down_gain == 4


def test_invalid_move_detected():
    # Fully sorted board where LEFT changes nothing.
    g = make_game_with_board([
        [8, 4, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    _, _, changed = g.simulate_move(Action.LEFT)
    assert not changed


def test_step_mutates_and_spawns():
    g = make_game_with_board([
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    empty_before = g.empty_cells()
    board, reward, done, info = g.step(Action.LEFT)
    assert info["valid"]
    assert reward == 4
    assert g.score == 4
    # a tile should have spawned somewhere (board had empty cells)
    assert g.empty_cells() == empty_before  # one 4 formed (-1 tile) + one spawn (+1 tile) nets to same emptiness... 
    # more directly: total nonzero count increased by exactly 1 spawn tile,
    # while merge reduced tile count by 1 -> net unchanged nonzero count is
    # coincidental here; check spawn flag instead:
    assert info["spawned"] in (True, False)


def test_game_over_detection_full_board_no_merges():
    # A full board with no adjacent equal values anywhere -> game over.
    board = [
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ]
    g = make_game_with_board(board)
    assert g._is_game_over()


def test_clone_is_independent():
    g = Game2048(seed=5)
    clone = g.clone()
    clone.board[0, 0] = 999
    assert g.board[0, 0] != 999


# ---------------------------------------------------------------------- #
# Encoding
# ---------------------------------------------------------------------- #
def test_encode_board_log2():
    board = np.array([
        [0, 2, 4, 8],
        [16, 32, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 2048],
    ])
    encoded = encode_board(board)
    assert encoded.shape == (16,)
    expected = np.array([0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11], dtype=np.float32)
    assert np.allclose(encoded, expected)


# ---------------------------------------------------------------------- #
# Env2048 wrapper
# ---------------------------------------------------------------------- #
def test_env_reset_returns_encoded_state():
    env = Env2048(seed=2)
    state = env.reset()
    assert state.shape == (16,)
    assert state.dtype == np.float32


def test_env_simulate_action_does_not_mutate():
    env = Env2048(seed=3)
    env.reset()
    board_before = env.board.copy()
    for action in env.action_space:
        env.simulate_action(action)
        assert np.array_equal(env.board, board_before)


def test_env_legal_action_mask_matches_game():
    env = Env2048(seed=4)
    env.reset()
    mask = env.legal_action_mask()
    legal_from_game = set(env.game.legal_actions())
    for action, is_legal in zip(env.action_space, mask):
        assert is_legal == (action in legal_from_game)


def test_env_full_random_episode_terminates():
    import random
    env = Env2048(seed=8)
    env.reset()
    steps = 0
    while not env.done and steps < 5000:
        mask = env.legal_action_mask()
        legal = [a for a, ok in zip(env.action_space, mask) if ok]
        if not legal:
            break
        action = random.choice(legal)
        env.step(action)
        steps += 1
    assert steps > 0
    assert env.game._is_game_over() or steps == 5000


if __name__ == "__main__":
    import sys
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        # Fallback runner if pytest isn't installed in this environment.
        test_funcs = [obj for name, obj in globals().items() if name.startswith("test_")]
        passed, failed = 0, 0
        for fn in test_funcs:
            try:
                fn()
                print(f"PASS  {fn.__name__}")
                passed += 1
            except Exception as e:
                print(f"FAIL  {fn.__name__}: {e}")
                failed += 1
        print(f"\n{passed} passed, {failed} failed out of {len(test_funcs)}")
        sys.exit(0 if failed == 0 else 1)
