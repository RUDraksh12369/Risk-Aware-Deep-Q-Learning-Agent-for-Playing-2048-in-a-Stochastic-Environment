"""Pure Monte Carlo tree search agent for 2048 demos.

This package is intentionally separate from the RL code. It uses only the
game rules engine in ``game.game`` and does not train, load, or depend on a
neural policy.
"""

from mcts.agent import MCTS2048Agent, MCTSConfig, choose_action

__all__ = ["MCTS2048Agent", "MCTSConfig", "choose_action"]
