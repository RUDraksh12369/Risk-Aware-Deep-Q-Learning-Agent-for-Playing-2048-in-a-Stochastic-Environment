import numpy as np

from game.game import Action, Game2048
from mcts.agent import MCTS2048Agent, MCTSConfig, choose_action


def make_game_with_board(board_values):
    game = Game2048(seed=0)
    game.board = np.array(board_values, dtype=np.int64)
    game.score = 0
    game.done = False
    return game


def test_mcts_returns_legal_action_without_mutating_board():
    game = make_game_with_board([
        [2, 2, 0, 0],
        [4, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    before = game.board.copy()
    agent = MCTS2048Agent(MCTSConfig(simulations=12, rollout_depth=6, seed=1))

    action = agent.choose_action(game)

    assert action in game.legal_actions()
    assert np.array_equal(game.board, before)


def test_mcts_handles_nearly_terminal_board():
    game = make_game_with_board([
        [2, 4, 8, 16],
        [4, 8, 16, 32],
        [8, 16, 32, 64],
        [16, 32, 64, 64],
    ])
    agent = MCTS2048Agent(MCTSConfig(simulations=8, rollout_depth=4, seed=2))

    assert game.legal_actions()
    assert agent.choose_action(game) in game.legal_actions()


def test_default_choose_action_callable():
    game = Game2048(seed=3)

    action = choose_action(game)

    assert action in game.legal_actions()
