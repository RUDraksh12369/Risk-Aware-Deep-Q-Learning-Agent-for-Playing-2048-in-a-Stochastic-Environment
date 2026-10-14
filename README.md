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
python -m training.train --episodes 2000 --checkpoint models/dqn_2048_enhanced.pt
python -m training.evaluate --checkpoint models/dqn_2048_enhanced.pt --episodes 100
```

## Dueling DQN (separate from the original DQN)

The original DQN remains independently runnable with its existing entrypoints:

```bash
python -m training.train --episodes 2000 --checkpoint models/dqn_2048_enhanced.pt
python -m training.evaluate --checkpoint models/dqn_2048.pt --episodes 100
```

Fine-tune the separate Dueling DQN from the existing checkpoint (this does not
write to `models/dqn_2048.pt`):

```bash
python -m rl.duelingDQN train --pretrained models/dqn_2048.pt --episodes 100
```

Resume its separately saved checkpoint, evaluate it, or compare both models
using the same episode seeds:

```bash
python -m rl.duelingDQN train --resume models/dueling_dqn_2048.pt --episodes 100
python -m rl.duelingDQN evaluate --checkpoint models/dueling_dqn_2048.pt --episodes 100
python -m rl.duelingDQN compare --dqn-checkpoint models/dqn_2048.pt --dueling-checkpoint models/dueling_dqn_2048.pt --episodes 100
```

Dueling checkpoints are stored under `models/dueling_dqn_2048.pt`; Dueling
training histories, transfer reports, and evaluation results are stored under
`artifacts/dueling_dqn/`.  The transfer report explicitly records copied and
newly initialised tensors.  Replay data is absent from the existing DQN
checkpoint, but is included in new Dueling checkpoints to support resuming.

## Demo GUI

The pygame GUI is for presentation/demo use only. It renders the project's
existing `Game2048` engine and is not used by training or tests.

```bash
python -m gui.pygame_demo --mode human
python -m gui.pygame_demo --mode heuristic --delay-ms 180
python -m gui.pygame_demo --mode mcts --delay-ms 140 --mcts-simulations 200
python -m gui.pygame_demo --mode random --seed 42
```

For a trained agent demo, expose a callable that returns one of
`game.game.Action` or action ids `0..3`, then run:

```bash
python -m gui.pygame_demo --mode agent --agent inference.autoplay:choose_action
```

`training.train` uses a masked Dueling Double DQN: the MLP receives the 16
log2 tile exponents, experience replay decorrelates updates, score rewards are
scaled for stable Q-targets, and the target network is softly synchronized.
Illegal moves are excluded in both behaviour and bootstrapping, which is
important because the environment otherwise permits them. Training runs for
the requested number of episodes. A checkpoint is saved after training (and
every 250 episodes by default).

The GUI automatically prefers `models/dqn_2048_enhanced.pt` when it exists,
falling back to the original `models/dqn_2048.pt` checkpoint. Set
`DQN_CHECKPOINT` to select a specific saved model.

### Separate Double DQN training

`rl/doubledqn.py` provides `DoubleDQNAgent` and the Double DQN next-action
calculation. It reuses the existing DQN network, replay buffer, learning loop,
and evaluation interface. The online network selects the best legal next
action and the target network evaluates it; terminal states do not bootstrap.
The original `DQNAgent` and its `double_dqn=False` standard-DQN option remain
available. Algorithm tests live in `tests/test_doubledqn.py`.

The dedicated trainer initializes fresh weights; it does not load the saved
DQN checkpoint. Its default output is `models/doubledqn_2048.pt`.

```bash
# Short CPU smoke run with learning enabled
python -m training.train_doubledqn --episodes 1 --max-steps-per-episode 32 --batch-size 4 --min-replay-size 4 --device cpu --checkpoint models/doubledqn_smoke.pt

# Full training, when ready
python -m training.train_doubledqn --episodes 2000 --checkpoint models/doubledqn_2048.pt

# Evaluate using the existing evaluator
python -m training.evaluate --checkpoint models/doubledqn_2048.pt --episodes 100
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
- [x] Step 3: Pure Monte Carlo Tree Search demo agent
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
