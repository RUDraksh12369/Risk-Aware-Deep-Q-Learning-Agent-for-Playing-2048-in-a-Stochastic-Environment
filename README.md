# Risk-Aware Deep Reinforcement Learning for Long-Horizon Decision Making under Stochastic Uncertainty
### An Empirical Study Using the 2048 Game Environment

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running tests

```bash
pytest -q
# or, if pytest isn't available:
PYTHONPATH=. python3 tests/test_game.py
```

## Quick smoke tests

```bash
python -m game.game          # pure mechanics demo (a few scripted moves)
python -m game.environment   # RL-wrapper demo (one full random episode)
```

## Repository structure

```
2048-risk-aware-rl/
├── game/
│   ├── game.py          # pure 2048 rules engine (board, moves, merges, spawn, game-over)
│   └── environment.py   # RL-facing wrapper: encoding, reward, Gym-like step/reset, action simulation
├── rl/
│   ├── dqn.py            # DQN network (MLP)
│   ├── replay_buffer.py  # experience replay
│   ├── agent.py           # epsilon-greedy agent, training loop glue
│   └── risk.py            # Board Risk Index + action-specific risk (Section 9-12)
├── gui/
│   ├── pygame_ui.py       # human-play + AI autoplay rendering
│   └── visualizer.py      # Q-value / risk analysis overlay
├── training/
│   ├── train.py           # training entrypoint (Standard DQN / Risk-Aware DQN)
│   └── evaluate.py         # batch evaluation across many episodes, metrics (Section 22)
├── inference/
│   └── autoplay.py         # load trained_model.pth, run/greedy-play, no learning
├── models/                 # saved .pth checkpoints
├── tests/
│   └── test_game.py         # unit tests for game mechanics + environment
└── requirements.txt
```

## Status

- [x] Step 1: 2048 environment (`game/game.py`, `game/environment.py`) + unit tests
- [ ] Step 2: Random & Heuristic baselines
- [ ] Step 3: Expectimax baseline
- [ ] Step 4: Evaluation framework (metrics harness)
- [ ] Step 5: Standard DQN (MLP) + replay buffer + target network + training loop
- [ ] Step 6: Board Risk Index + action-specific risk module
- [ ] Step 7: Risk-Aware DQN (risk-adjusted action selection / reward shaping)
- [ ] Step 8: Ablation study
- [ ] Step 9: Risk-sensitivity (λ sweep) experiments
- [ ] Step 10: Robustness / generalization experiments
- [ ] Step 11: Pygame GUI (human mode, autoplay, analysis overlay)
- [ ] Step 12: Optional CNN-DQN comparison
- [ ] Step 13: Final report / presentation

See project context doc for full research questions, risk formulation, and
the 10–12 week plan / group division of labor.
