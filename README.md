# yasta-bot

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

### Requirements

- Python 3.7+
- numpy

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
- **Exact**: Follows specification exactly with no deviations
- **Zero external dependencies**: Only numpy, queue/deque, and base Python

### Implementation Notes

- BFS operations use deterministic neighbor ordering: Up, Right, Down, Left
- All cloning operations create full deep copies
- No randomness except via `utils.set_seed()`
- Simultaneous movement with exact collision logic
- Rewards are always 0 during gameplay (for RL training)
