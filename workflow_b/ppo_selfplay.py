"""
PPO Self-Play Training Loop.
Part of Workflow B - RL Training Pipeline.

Implements PPO with self-play and periodic opponent snapshotting.
"""

import sys
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow_b.policy import TronPolicy
from workflow_b.selfplay_env import SelfPlayWrapper
from workflow_b.buffer import PPOBuffer


def train_ppo_selfplay(
    initial_model_path: str,
    output_dir: str = "models",
    steps_per_iteration: int = 8192,
    ppo_epochs: int = 4,
    mini_batch_size: int = 512,
    clip_ratio: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    lr: float = 3e-4,
    max_grad_norm: float = 0.5,
    opponent_update_freq: int = 5,
    total_iterations: int = 100,
    save_freq: int = 1
) -> None:
    """
    Train policy using PPO with self-play.
    
    Args:
        initial_model_path: Path to initial model (e.g., BC model)
        output_dir: Directory to save models
        steps_per_iteration: Steps to collect per iteration
        ppo_epochs: Number of PPO update epochs per iteration
        mini_batch_size: Mini-batch size for PPO updates
        clip_ratio: PPO clipping parameter
        value_loss_coef: Coefficient for value loss
        entropy_coef: Coefficient for entropy bonus
        lr: Learning rate
        max_grad_norm: Maximum gradient norm for clipping
        opponent_update_freq: Iterations between opponent updates
        total_iterations: Total number of iterations
        save_freq: Save model every N iterations
    """
    print("Initializing PPO self-play training...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sp_archive'), exist_ok=True)
    
    # Load initial policy
    print(f"Loading initial model from {initial_model_path}...")
    current_policy = TronPolicy()
    current_policy.load_state_dict(torch.load(initial_model_path))
    
    # Create opponent policy (initial snapshot)
    opponent_policy = TronPolicy()
    opponent_policy.load_state_dict(torch.load(initial_model_path))
    opponent_policy.eval()
    
    # Create self-play environment
    env = SelfPlayWrapper(current_policy, opponent_policy)
    
    # Create buffer
    buffer = PPOBuffer(size=steps_per_iteration, obs_shape=(8, 18, 20))
    
    # Optimizer
    optimizer = optim.Adam(current_policy.parameters(), lr=lr)
    
    # Training loop
    print(f"\nStarting training for {total_iterations} iterations...")
    
    for iteration in range(total_iterations):
        print(f"\n=== Iteration {iteration + 1}/{total_iterations} ===")
        
        # Collect experience
        print("Collecting experience...")
        current_policy.eval()
        
        episodes = 0
        total_reward = 0.0
        wins = 0
        losses = 0
        draws = 0
        
        while buffer.ptr < steps_per_iteration:
            obs = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                # Get action from policy
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                
                with torch.no_grad():
                    policy_logits, value = current_policy(obs_tensor)
                    value = value.item()
                    
                    # Sample action
                    probs = torch.softmax(policy_logits, dim=1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    logprob = dist.log_prob(action).item()
                    action = action.item()
                
                # Step environment
                next_obs, reward, done, info = env.step(action)
                
                # Store in buffer
                buffer.store(obs, action, reward, value, logprob)
                
                obs = next_obs
                episode_reward += reward
                
                # Check if buffer is full
                if buffer.ptr >= steps_per_iteration:
                    break
            
            # Finish trajectory
            if done:
                buffer.finish_path(last_value=0.0)
            else:
                # Bootstrap from value function
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                with torch.no_grad():
                    _, value = current_policy(obs_tensor)
                    value = value.item()
                buffer.finish_path(last_value=value)
            
            # Track episode stats
            episodes += 1
            total_reward += episode_reward
            
            if episode_reward > 0.5:
                wins += 1
            elif episode_reward < -0.5:
                losses += 1
            else:
                draws += 1
        
        avg_reward = total_reward / episodes if episodes > 0 else 0.0
        print(f"Collected {buffer.ptr} steps from {episodes} episodes")
        print(f"Avg reward: {avg_reward:.3f}, W/L/D: {wins}/{losses}/{draws}")
        
        # Get data from buffer
        data = buffer.get()
        
        # PPO update
        print("Running PPO update...")
        current_policy.train()
        
        dataset_size = len(data['obs'])
        indices = np.arange(dataset_size)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        update_count = 0
        
        for epoch in range(ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, mini_batch_size):
                end = min(start + mini_batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = data['obs'][batch_indices]
                batch_actions = data['actions'][batch_indices]
                batch_old_logprobs = data['logprobs'][batch_indices]
                batch_advantages = data['advantages'][batch_indices]
                batch_returns = data['returns'][batch_indices]
                
                # Forward pass
                policy_logits, values = current_policy(batch_obs)
                values = values.squeeze(-1)
                
                # Compute log probs
                probs = torch.softmax(policy_logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio and clipped objective
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
                
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Total loss
                loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(current_policy.parameters(), max_grad_norm)
                optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_count += 1
        
        avg_policy_loss = total_policy_loss / update_count if update_count > 0 else 0.0
        avg_value_loss = total_value_loss / update_count if update_count > 0 else 0.0
        avg_entropy = total_entropy / update_count if update_count > 0 else 0.0
        
        print(f"PPO update complete - Policy loss: {avg_policy_loss:.4f}, Value loss: {avg_value_loss:.4f}, Entropy: {avg_entropy:.4f}")
        
        # Update opponent periodically
        if (iteration + 1) % opponent_update_freq == 0:
            print("Updating opponent policy...")
            opponent_policy = TronPolicy()
            opponent_policy.load_state_dict(current_policy.state_dict())
            opponent_policy.eval()
            env.update_opponent(opponent_policy)
            
            # Save snapshot
            snapshot_path = os.path.join(output_dir, 'sp_archive', f'opponent_iter_{iteration + 1}.pth')
            torch.save(opponent_policy.state_dict(), snapshot_path)
            print(f"Opponent snapshot saved to {snapshot_path}")
        
        # Save latest model
        if (iteration + 1) % save_freq == 0:
            latest_path = os.path.join(output_dir, 'policy_ppo_latest.pth')
            torch.save(current_policy.state_dict(), latest_path)
            print(f"Model saved to {latest_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO with self-play")
    parser.add_argument("--initial_model", type=str, default="models/policy_bc.pth",
                        help="Initial model path (default: models/policy_bc.pth)")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Output directory (default: models)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Total iterations (default: 100)")
    parser.add_argument("--steps_per_iteration", type=int, default=8192,
                        help="Steps per iteration (default: 8192)")
    
    args = parser.parse_args()
    
    train_ppo_selfplay(
        initial_model_path=args.initial_model,
        output_dir=args.output_dir,
        total_iterations=args.iterations,
        steps_per_iteration=args.steps_per_iteration
    )
