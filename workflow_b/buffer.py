"""
PPO Replay Buffer with GAE-Lambda.
Part of Workflow B - RL Training Pipeline.

Stores experience and computes advantages using Generalized Advantage Estimation.
"""

import numpy as np
import torch
from typing import Optional


class PPOBuffer:
    """
    Replay buffer for PPO with GAE-Lambda advantage estimation.
    
    Stores trajectories and computes advantages for policy optimization.
    """
    
    def __init__(self, size: int, obs_shape: tuple, gamma: float = 0.99, lam: float = 0.95):
        """
        Initialize PPO buffer.
        
        Args:
            size: Maximum number of transitions to store
            obs_shape: Shape of observations (8, 18, 20)
            gamma: Discount factor
            lam: GAE-Lambda parameter
        """
        self.size = size
        self.gamma = gamma
        self.lam = lam
        
        # Storage arrays
        self.obs = np.zeros((size,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.logprobs = np.zeros(size, dtype=np.float32)
        
        # Computed arrays
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        
        # Tracking
        self.ptr = 0
        self.path_start_idx = 0
    
    def store(self, obs: np.ndarray, action: int, reward: float, 
              value: float, logprob: float) -> None:
        """
        Store a single transition.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            logprob: Log probability of action
        """
        assert self.ptr < self.size, "Buffer overflow"
        
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.logprobs[self.ptr] = logprob
        
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0.0) -> None:
        """
        Finish a trajectory and compute advantages using GAE-Lambda.
        
        Args:
            last_value: Bootstrap value for last state (0 if terminal)
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # Compute TD residuals
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        
        # Compute GAE advantages
        advantages = np.zeros_like(deltas)
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.lam * gae
            advantages[t] = gae
        
        self.advantages[path_slice] = advantages
        
        # Compute returns (advantages + values)
        self.returns[path_slice] = advantages + self.values[path_slice]
        
        # Update path start
        self.path_start_idx = self.ptr
    
    def get(self) -> dict[str, torch.Tensor]:
        """
        Get all stored data and reset buffer.
        
        Returns:
            Dictionary containing:
                - obs: Observations tensor
                - actions: Actions tensor
                - logprobs: Log probabilities tensor
                - advantages: Advantages tensor (normalized)
                - returns: Returns tensor
        """
        assert self.ptr > 0, "Buffer is empty"
        
        # Get data slice
        data = {
            'obs': torch.from_numpy(self.obs[:self.ptr]).float(),
            'actions': torch.from_numpy(self.actions[:self.ptr]).long(),
            'logprobs': torch.from_numpy(self.logprobs[:self.ptr]).float(),
            'advantages': torch.from_numpy(self.advantages[:self.ptr]).float(),
            'returns': torch.from_numpy(self.returns[:self.ptr]).float()
        }
        
        # Normalize advantages
        data['advantages'] = (data['advantages'] - data['advantages'].mean()) / (data['advantages'].std() + 1e-8)
        
        # Reset buffer
        self.ptr = 0
        self.path_start_idx = 0
        
        return data
