# Workflow B - RL Training Pipeline

This directory contains the complete reinforcement learning training pipeline for the Tron bot.

## Overview

The pipeline consists of three main stages:

1. **Behavior Cloning (BC)** - Learn from heuristic bot demonstrations
2. **PPO Self-Play** - Improve through self-play with PPO algorithm
3. **Agent Export** - Create competition-ready agent

## Directory Structure

```
workflow_b/
├── policy.py              # Residual CNN architecture (AlphaZero-style)
├── bc_dataset.py          # Generate BC dataset from heuristic bot
├── bc_train.py            # Train BC model
├── selfplay_env.py        # Self-play environment wrapper
├── buffer.py              # PPO replay buffer with GAE
├── ppo_selfplay.py        # PPO self-play training loop
├── train_all.py           # Master training script
├── eval.py                # Evaluation against various opponents
└── make_agent_export.py   # Export competition agent
```

## Quick Start

### Option 1: Run Complete Pipeline

```bash
python workflow_b/train_all.py
```

This will:
1. Generate BC dataset (200 games, ~2000 samples)
2. Train BC model (30 epochs)
3. Train PPO with self-play (100+ iterations)

Models are saved to `models/` directory.

### Option 2: Run Individual Steps

**Step 1: Generate BC Dataset**

```bash
python workflow_b/bc_dataset.py --num_games 200 --output models/bc_dataset.npz
```

**Step 2: Train BC Model**

```bash
python workflow_b/bc_train.py --dataset models/bc_dataset.npz --epochs 30
```

**Step 3: Train PPO Self-Play**

```bash
python workflow_b/ppo_selfplay.py --initial_model models/policy_bc.pth --iterations 100
```

## Evaluation

Evaluate any trained model:

```bash
python workflow_b/eval.py --model models/policy_ppo_latest.pth --games 200
```

This tests the model against:
- Heuristic bot
- Random bot
- Self (mirror match)

## Export for Competition

Create a standalone agent.py file:

```bash
python workflow_b/make_agent_export.py --model models/policy_ppo_latest.pth --output agent.py
```

This generates:
- `agent.py` - Competition agent code
- `agent_weights.pth` - Model weights

## Architecture Details

### Residual CNN Policy (policy.py)

- **Input**: (8, 18, 20) observation tensor
- **Conv Stem**: 8 → 64 channels
- **Residual Tower**: 10 residual blocks
- **Policy Head**: 4 action logits
- **Value Head**: Scalar value in [-1, 1]

### Training Hyperparameters

**Behavior Cloning:**
- Epochs: 30
- Batch size: 128
- Learning rate: 1e-3
- Optimizer: Adam

**PPO Self-Play:**
- Steps per iteration: 8192
- PPO epochs: 4
- Mini-batch size: 512
- Clip ratio: 0.2
- Value loss coefficient: 0.5
- Entropy coefficient: 0.01
- Learning rate: 3e-4
- Opponent update frequency: 5 iterations

### GAE-Lambda (buffer.py)

- Gamma (discount): 0.99
- Lambda: 0.95
- Normalized advantages

## Output Files

Training produces the following files:

```
models/
├── bc_dataset.npz           # BC training data
├── bc_loss_curve.png        # BC training metrics
├── policy_bc.pth            # BC model weights
├── policy_ppo_latest.pth    # Latest PPO model
└── sp_archive/              # Opponent snapshots
    └── opponent_iter_*.pth
```

## Requirements

- Python 3.7+
- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0 (for training curves)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- All training is CPU-only (tournament constraint)
- Models are deterministic (no dropout, fixed seeds)
- PPO training can be interrupted with Ctrl-C (model is saved)
- Opponent is updated every 5 iterations for diversity
- Terminal rewards only (+1 win, 0 draw, -1 loss)

## Performance Expectations

**BC Model:**
- Should achieve ~100% accuracy on heuristic demonstrations
- Mirrors heuristic bot behavior initially

**PPO Model:**
- Improves through self-play
- Win rate vs heuristic varies (often starts with draws)
- Longer training (300-800 iterations) recommended for best results

## Troubleshooting

**Out of Memory:**
- Reduce `steps_per_iteration` (try 4096)
- Reduce `mini_batch_size` (try 256)

**Training Too Slow:**
- BC is fast (~2-3 minutes)
- PPO is slow (~1-2 hours for 100 iterations)
- Use smaller iteration count for testing

**Model Not Learning:**
- Ensure BC dataset has enough samples (>1000)
- Check BC accuracy (should be >90%)
- PPO may need more iterations to show improvement
