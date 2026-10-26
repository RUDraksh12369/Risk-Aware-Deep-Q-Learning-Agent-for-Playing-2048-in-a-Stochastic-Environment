"""Load a trained DQN checkpoint and expose a GUI-compatible greedy policy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from game.environment import encode_board
from game.game import Action, Game2048
from rl.agent import DQNAgent

_agent: Optional[DQNAgent] = None
_agent_key: tuple[Path, Optional[str]] | None = None
ENHANCED_CHECKPOINT = Path("models/dqn_2048_enhanced.pt")
BASELINE_CHECKPOINT = Path("models/dqn_2048.pt")


def load_policy(checkpoint: str | Path | None = None, device: Optional[str] = None) -> DQNAgent:
    """Load a checkpoint once and return its DQN agent."""
    global _agent, _agent_key
    if checkpoint is None:
        path = ENHANCED_CHECKPOINT if ENHANCED_CHECKPOINT.is_file() else BASELINE_CHECKPOINT
    else:
        path = Path(checkpoint)
    key = (path.resolve(), device)
    if _agent is None or _agent_key != key:
        if not path.is_file():
            raise FileNotFoundError(f"DQN checkpoint not found: {path}. Train one with `python -m training.train`.")
        _agent = DQNAgent.load(path, device=device)
        _agent_key = key
    return _agent


def choose_action(game: Game2048) -> Action:
    """Greedy callable for ``pygame_demo --agent inference.autoplay:choose_action``.

    Set ``DQN_CHECKPOINT`` to choose a checkpoint other than the default.
    """
    agent = load_policy(os.environ.get("DQN_CHECKPOINT"))
    legal = game.legal_actions()
    if not legal:
        raise ValueError("DQN cannot choose an action from a terminal board.")
    mask = [action in legal for action in Action]
    return Action(agent.select_action(encode_board(game.board), mask, epsilon=0.0, game=game))
