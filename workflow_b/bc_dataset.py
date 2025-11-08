"""
Behavior Cloning Dataset Generation.
Part of Workflow B - RL Training Pipeline.

Generates training data by running the heuristic bot against itself.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import TronEnv
from heuristic_bot import choose_action


def generate_bc_data(num_games: int, output_path: str) -> None:
    """
    Generate behavior cloning dataset from heuristic bot self-play.
    
    Simulates random games using the heuristic bot playing both sides.
    Collects observations and actions from both perspectives.
    
    Args:
        num_games: Number of games to simulate
        output_path: Path to save the .npz file
    """
    print(f"Generating BC dataset with {num_games} games...")
    
    all_obs = []
    all_actions = []
    
    for game_idx in range(num_games):
        if (game_idx + 1) % 100 == 0:
            print(f"  Game {game_idx + 1}/{num_games}")
        
        # Create environment with random seed for variety
        env = TronEnv(seed=game_idx)
        obs_p1 = env.reset(who='p1')
        obs_p2 = env.reset(who='p2')
        
        done = False
        
        while not done:
            # Get actions from heuristic bot for both players
            action_p1 = choose_action(env, 'p1')
            action_p2 = choose_action(env, 'p2')
            
            # Store observations and actions from both perspectives
            all_obs.append(obs_p1.copy())
            all_actions.append(action_p1)
            
            all_obs.append(obs_p2.copy())
            all_actions.append(action_p2)
            
            # Step environment
            obs_p1, _, _, done, info = env.step(action_p1, action_p2, who='p1')
            
            # Get observation from p2's perspective if game is not done
            if not done:
                obs_p2 = env._get_obs('p2')
    
    # Convert to numpy arrays
    obs_array = np.array(all_obs, dtype=np.float32)
    actions_array = np.array(all_actions, dtype=np.int64)
    
    print(f"\nGenerated {len(all_obs)} samples")
    print(f"Observation shape: {obs_array.shape}")
    print(f"Actions shape: {actions_array.shape}")
    
    # Save to compressed npz file
    np.savez_compressed(
        output_path,
        obs=obs_array,
        actions=actions_array
    )
    
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate BC dataset")
    parser.add_argument("--num_games", type=int, default=200,
                        help="Number of games to simulate (default: 200)")
    parser.add_argument("--output", type=str, default="models/bc_dataset.npz",
                        help="Output path for dataset (default: models/bc_dataset.npz)")
    
    args = parser.parse_args()
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    generate_bc_data(args.num_games, args.output)
