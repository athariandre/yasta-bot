"""
Self-Play Environment Wrapper for PPO.
Part of Workflow B - RL Training Pipeline.

Wraps TronEnv for self-play training with opponent snapshots.
"""

import sys
import os
import copy
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import TronEnv
from workflow_b.policy import TronPolicy


class SelfPlayWrapper:
    """
    Self-play wrapper for PPO training.
    
    Maintains two policies:
    - current policy (being trained)
    - opponent policy (frozen snapshot)
    
    Alternates perspectives and returns terminal rewards only.
    """
    
    def __init__(self, current_policy: TronPolicy, opponent_policy: TronPolicy):
        """
        Initialize self-play wrapper.
        
        Args:
            current_policy: Current policy being trained
            opponent_policy: Opponent policy (frozen snapshot)
        """
        self.current_policy = current_policy
        self.opponent_policy = opponent_policy
        self.env = TronEnv()
        self.game_count = 0
    
    def reset(self) -> np.ndarray:
        """
        Reset environment and alternate perspective.
        
        Returns:
            Initial observation for current policy
        """
        # Alternate who plays as p1 vs p2
        self.i_am_p1 = (self.game_count % 2 == 0)
        self.game_count += 1
        
        # Reset environment from appropriate perspective
        who = 'p1' if self.i_am_p1 else 'p2'
        obs = self.env.reset(who=who)
        
        return obs
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step with current policy and opponent policy.
        
        Args:
            action: Action chosen by current policy
            
        Returns:
            Tuple of (next_obs, reward, done, info):
                - next_obs: Next observation for current policy
                - reward: Terminal reward (+1 win, 0 draw, -1 loss)
                - done: Whether episode is over
                - info: Additional info dict
        """
        # Get opponent action
        opp_obs = self.env._get_obs('p2' if self.i_am_p1 else 'p1')
        opp_action = self._get_opponent_action(opp_obs)
        
        # Determine action order based on perspective
        if self.i_am_p1:
            action_p1 = action
            action_p2 = opp_action
            who = 'p1'
        else:
            action_p1 = opp_action
            action_p2 = action
            who = 'p2'
        
        # Step environment
        next_obs, _, _, done, info = self.env.step(action_p1, action_p2, who=who)
        
        # Compute terminal reward
        if done:
            my_dead = info['p1_dead'] if self.i_am_p1 else info['p2_dead']
            opp_dead = info['p2_dead'] if self.i_am_p1 else info['p1_dead']
            
            if my_dead and opp_dead:
                # Both died (draw)
                reward = 0.0
            elif my_dead:
                # I died (loss)
                reward = -1.0
            else:
                # Opponent died (win)
                reward = 1.0
        else:
            # No reward during episode
            reward = 0.0
        
        return next_obs, reward, done, info
    
    def _get_opponent_action(self, obs: np.ndarray) -> int:
        """
        Get action from opponent policy.
        
        Args:
            obs: Observation for opponent
            
        Returns:
            Action index
        """
        self.opponent_policy.eval()
        
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
            policy_logits, _ = self.opponent_policy(obs_tensor)
            
            # Sample from policy
            probs = torch.softmax(policy_logits, dim=1)
            action = torch.multinomial(probs, num_samples=1).item()
        
        return action
    
    def update_opponent(self, new_opponent_policy: TronPolicy) -> None:
        """
        Update opponent policy with a new snapshot.
        
        Args:
            new_opponent_policy: New opponent policy to use
        """
        self.opponent_policy = new_opponent_policy
