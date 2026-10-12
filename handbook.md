# Project Handbook - 2048 Risk-Aware RL

## Purpose and current position

This repository is building a research-style 2048 agent for studying
long-horizon decisions under stochastic tile spawns. The foundation, planning
baseline, DQN baseline, training workflow, inference adapter, and presentation
GUI are implemented. The advertised *risk-aware* part of the study is not yet
implemented: there is no `rl/risk.py`, risk-adjusted policy, risk-shaped
reward, experiment suite, or final comparative report.

In short: the project has a working and tested **standard DQN baseline** and
an **MCTS comparison/demo agent**; it is ready for proper training runs and
the risk-aware research phase.

## What has been completed

| Area | Status | What is present |
| --- | --- | --- |
| 2048 rules engine | Complete | Tile spawning, directional movement, merges, score tracking, terminal detection, cloning, move simulation, and legal-action discovery. |
| RL environment | Complete | Gym-like `reset`/`step`, log2 state encoding, rewards, legal-action masks, and non-mutating afterstate simulation. |
| Tests | Complete for implemented features | 25 passing tests cover game rules, environment behaviour, MCTS behaviour, replay, DQN action masking/learning, and evaluation metrics. |
| MCTS baseline/demo | Complete | Stochastic rollouts, UCB action selection, simulated tile spawns, and board heuristics. |
| DQN baseline | Complete | Dueling MLP, Double-DQN targets, replay buffer, target network, reward scaling, gradient clipping, checkpoint save/load, and legal-action masking. |
| Training and evaluation | Complete | CLI trainer with epsilon schedule/checkpoints and deterministic greedy batch evaluation. |
| Inference and GUI | Complete | Checkpoint-loading action callable and Pygame modes for human, random, heuristic, MCTS, and supplied-agent play. |
| Saved artefacts | Present | A trained checkpoint is included in `models/dqn_2048.pt`; the current checkpoint records 119,106 learning updates. |

## How the implemented system works

### 1. Game and environment

`game/game.py` is the source of truth for 2048 mechanics. A move compresses
and merges tiles once per row/column, adds the merge value to the score, and
then spawns a random tile (90% 2, 10% 4 by default). It also exposes
`simulate_move`, which permits agents to inspect a move without changing the
live game.

`game/environment.py` keeps RL concerns separate from game rules. It converts
the 4x4 board into 16 float32 log2 tile exponents (`0 -> 0`, `2 -> 1`, `4 -> 2`,
and so on), returns score gain as the default reward, and exposes a boolean
legal-action mask. Optional illegal-move, game-over, and custom reward hooks
are already available for later experiments.

### 2. Planning baseline

`mcts/agent.py` implements an independent Monte Carlo Tree Search player. For
each legal move it simulates the random spawn, uses UCB to balance exploration
and exploitation, performs bounded rollouts, and backs up accumulated return.
Rollouts prefer immediate gains, empty cells, corner placement of the maximum
tile, and monotonic rows/columns. It is a useful playable baseline, but it is
not a risk-aware algorithm or an optimal 2048 solver.

### 3. DQN baseline

The learning agent is split into three parts:

- `rl/dqn.py` defines a 16-input, two-hidden-layer (256/256) MLP. It
  normalises the log2 encoding and uses a dueling value/advantage head.
- `rl/replay_buffer.py` stores fixed-capacity transitions and samples random
  mini-batches to decorrelate updates.
- `rl/agent.py` performs epsilon-greedy action selection, never chooses a
  masked/illegal move, trains with Smooth L1 loss, clips gradients, and keeps a
  softly updated target network.

The update is Double DQN: the online network chooses the next legal action and
the target network evaluates it. Terminal or no-legal-action transitions do
not bootstrap. Raw game rewards are multiplied by `0.01` only when written to
replay, keeping reported scores in normal 2048 units while improving Q-target
scale.

### 4. Running, saving, and evaluating

`training/train.py` creates an environment and agent, linearly decays epsilon
from 1.0 to 0.05 over 300,000 steps, records per-episode score/tile/loss data,
and saves checkpoints every 250 episodes plus at the end. Defaults include a
2,000-transition warm-up, batch size 128, gamma 0.995, and soft target update
tau 0.005.

`training/evaluate.py` loads a checkpoint and plays greedy episodes with fresh,
seeded environments. It reports mean/median/best score, mean maximum tile, and
rates of reaching tiles 128, 256, and 512. `inference/autoplay.py` provides the
same loaded policy to the GUI.

## Verification completed

The current workspace was checked with:

```powershell
pytest -q
```

Result: **25 passed**. A five-episode CPU smoke evaluation of the bundled
`models/dqn_2048.pt` also ran successfully (mean score 1112.8; 3/5 games
reached tile 128). This is a runtime sanity check only, not a statistically
reliable benchmark.

## Remaining work

The following research milestones remain before the repository matches its
"risk-aware" title:

1. Add a board-risk index and action-specific afterstate risk module.
2. Implement risk-aware DQN action selection and/or risk-shaped rewards, with
   a configurable risk coefficient (lambda).
3. Add random and explicit heuristic baselines to the evaluation harness
   (the GUI has simple random/heuristic play, but no recorded benchmark suite).
4. Run repeatable, multi-seed training for the standard and risk-aware agents;
   retain histories, checkpoints, and evaluation tables.
5. Perform ablations, lambda sensitivity, and robustness/generalisation experiments.
6. Produce plots, conclusions, and the final report/presentation. A CNN-DQN
   comparison is optional.

## Practical commands

```powershell
# Install dependencies in an activated virtual environment
pip install -r requirements.txt

# Validate the implementation
pytest -q

# Train a baseline checkpoint
python -m training.train --episodes 2000 --checkpoint models/dqn_2048_enhanced.pt

# Evaluate a trained checkpoint
python -m training.evaluate --checkpoint models/dqn_2048_enhanced.pt --episodes 100

# Demo the agent or MCTS
python -m gui.pygame_demo --mode agent --agent inference.autoplay:choose_action
python -m gui.pygame_demo --mode mcts --mcts-simulations 200
```

## Important caveats

- The included checkpoint is evidence that the pipeline runs; it should not be
  presented as the final trained or risk-aware result without multi-seed
  evaluation.
- The trainer prints and stores histories inside checkpoint metadata, but does
  not yet write CSV files or plots.
- The README status checklist is stale: it marks DQN, evaluation, and GUI
  items incomplete even though their implementations and tests are now in the
  repository.
