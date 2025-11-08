# yasta-bot

Tron AI bot with complete reinforcement learning training pipeline.

## Overview

This repository implements a complete Tron game environment and RL training pipeline:

- **Workflow A**: Core game engine, features, and heuristic bot
- **Workflow B**: Behavior cloning and PPO self-play training

## Quick Start

### Train a Bot

```bash
python workflow_b/train_all.py
```

This runs the complete pipeline:
1. Generate behavior cloning dataset
2. Train BC model
3. Train PPO with self-play

### Evaluate a Model

```bash
python workflow_b/eval.py --model models/policy_ppo_latest.pth --games 100
```

### Export for Competition

```bash
python workflow_b/make_agent_export.py --model models/policy_ppo_latest.pth
```

## Workflow A - Core Engine Implementation

This repository implements the core Tron game environment for reinforcement learning training (Task 1 / Workflow A).

### Modules Implemented

1. **utils.py** - Utility functions and logger
   - `set_seed(seed)` - Deterministic RNG seeding
   - `Logger` class - Minimal logging

2. **env.py** - Core Tron game environment
   - `TronEnv` class with full game mechanics
   - 18x20 grid, simultaneous movement
   - Exact observation tensor format (8 channels)
   - Deep cloning support

3. **features.py** - Numpy-optimized game analysis
   - `reachable_area()` - BFS flood-fill
   - `voronoi_counts()` - Multi-source BFS
   - `is_tunnel()` - Corridor detection
   - `simulate_step()` - Non-mutating simulation

4. **heuristic_bot.py** - Deterministic expert policy
   - `choose_action()` - Strategic move selection
   - Area difference + tunnel penalty + cut bonus
   - Deterministic tie-breaking

## Workflow B - RL Training Pipeline

Complete reinforcement learning training pipeline with behavior cloning and PPO self-play.

### Components

- **policy.py** - Residual CNN architecture (AlphaZero-style)
- **bc_dataset.py** - Behavior cloning dataset generation
- **bc_train.py** - BC model training
- **selfplay_env.py** - Self-play environment wrapper
- **buffer.py** - PPO replay buffer with GAE
- **ppo_selfplay.py** - PPO self-play training loop
- **train_all.py** - Master training script
- **eval.py** - Model evaluation
- **make_agent_export.py** - Competition agent export

See [workflow_b/README.md](workflow_b/README.md) for detailed documentation.

### Requirements

- Python 3.7+
- numpy >= 1.20.0
- torch >= 1.9.0
- matplotlib >= 3.3.0

Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Run tests:
```bash
python test_workflow_a.py
```

Run example game:
```bash
python example_game.py
```

### Key Features

- **Deterministic**: All functions produce identical outputs given identical inputs
- **Fast**: Optimized numpy operations, O(rows × cols) complexity
- **CPU-Only**: Compatible with tournament constraints
- **Modular**: Clean separation between game engine and training

### Implementation Notes

- BFS operations use deterministic neighbor ordering: Up, Right, Down, Left
- All cloning operations create full deep copies
- No randomness except via `utils.set_seed()`
- Simultaneous movement with exact collision logic
- Rewards are always 0 during gameplay (terminal rewards for RL training)
