"""
Core Tron Game Environment for RL Training.
Part of Workflow A - Core Engine Implementation.

This module implements the TronEnv class with exact deterministic behavior
for reinforcement learning training.
"""

import numpy as np
import copy


class TronEnv:
    """
    Tron game environment with simultaneous movement.
    
    Grid coordinates: (row, col)
    Board size: 18 rows × 20 columns
    Two players: 'p1' and 'p2'
    """
    
    # Action constants: Up, Right, Down, Left
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    # Direction vectors in deterministic order: Up, Right, Down, Left
    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    def __init__(self, rows=18, cols=20, seed=None):
        """
        Initialize the Tron environment.
        
        Args:
            rows: Number of rows (default 18)
            cols: Number of columns (default 20)
            seed: Random seed for deterministic behavior (optional)
        """
        self.rows = rows
        self.cols = cols
        
        if seed is not None:
            np.random.seed(seed)
        
        # Grid: 0 = empty, 1 = p1 trail, 2 = p2 trail
        self.grid = np.zeros((rows, cols), dtype=np.int8)
        
        # Player positions and directions
        self.p1_pos = None
        self.p2_pos = None
        self.p1_dir = None
        self.p2_dir = None
        
        # Game state
        self.p1_dead = False
        self.p2_dead = False
        self.turn = 0
        
    def reset(self, start_mode="mirror", who="p1"):
        """
        Reset the environment to initial state.
        
        Args:
            start_mode: Starting mode (currently only "mirror" supported)
            who: Perspective for observation ('p1' or 'p2')
            
        Returns:
            obs: Initial observation tensor
        """
        # Reset grid
        self.grid.fill(0)
        
        # Set spawn positions
        # p1 at (rows//2, cols//4)
        # p2 at (rows//2, cols - cols//4)
        self.p1_pos = (self.rows // 2, self.cols // 4)
        self.p2_pos = (self.rows // 2, self.cols - self.cols // 4)
        
        # Initial directions (p1 faces right, p2 faces left)
        self.p1_dir = self.RIGHT
        self.p2_dir = self.LEFT
        
        # Mark initial positions on grid
        self.grid[self.p1_pos] = 1
        self.grid[self.p2_pos] = 2
        
        # Reset game state
        self.p1_dead = False
        self.p2_dead = False
        self.turn = 0
        
        # Return observation from specified player's perspective
        return self._get_obs(who)
    
    def step(self, action_my, action_opp, who="p1"):
        """
        Execute one step with simultaneous movement.
        
        Args:
            action_my: Action for 'p1' (0=Up, 1=Right, 2=Down, 3=Left)
            action_opp: Action for 'p2'
            who: Perspective for observation ('p1' or 'p2')
            
        Returns:
            obs: Observation tensor
            reward_my: Reward for p1 (always 0 during game)
            reward_opp: Reward for p2 (always 0 during game)
            done: True if game is over
            info: Dictionary with game state info
        """
        # Compute new positions BEFORE collision logic
        new_p1_pos = self._compute_new_position(self.p1_pos, action_my)
        new_p2_pos = self._compute_new_position(self.p2_pos, action_opp)
        
        # Update directions
        self.p1_dir = action_my
        self.p2_dir = action_opp
        
        # Check crashes
        p1_crashed = self._is_crash(new_p1_pos)
        p2_crashed = self._is_crash(new_p2_pos)
        
        # Head-on collision: both crash
        if new_p1_pos == new_p2_pos:
            p1_crashed = True
            p2_crashed = True
        
        # Update positions if not crashed
        if not p1_crashed:
            self.p1_pos = new_p1_pos
            self.grid[new_p1_pos] = 1
        else:
            self.p1_dead = True
        
        if not p2_crashed:
            self.p2_pos = new_p2_pos
            self.grid[new_p2_pos] = 2
        else:
            self.p2_dead = True
        
        # Increment turn counter
        self.turn += 1
        
        # Game is done if any player is dead
        done = self.p1_dead or self.p2_dead
        
        # Rewards are always 0 during game
        reward_my = 0
        reward_opp = 0
        
        # Build info dict
        info = {
            "p1_dead": self.p1_dead,
            "p2_dead": self.p2_dead,
            "turn": self.turn
        }
        
        # Get observation from specified player's perspective
        obs = self._get_obs(who)
        
        return obs, reward_my, reward_opp, done, info
    
    def legal_moves(self, who):
        """
        Get list of legal moves for a player.
        
        Args:
            who: 'p1' or 'p2'
            
        Returns:
            List of legal action indices
        """
        pos = self.p1_pos if who == 'p1' else self.p2_pos
        legal = []
        
        for action in range(4):
            new_pos = self._compute_new_position(pos, action)
            if not self._is_crash(new_pos):
                legal.append(action)
        
        return legal
    
    def clone(self):
        """
        Create a deep copy of the environment.
        
        Returns:
            A fully independent copy of this environment
        """
        # Create new environment instance
        cloned = TronEnv(self.rows, self.cols)
        
        # Deep copy grid
        cloned.grid = self.grid.copy()
        
        # Copy positions (tuples are immutable, safe to copy)
        cloned.p1_pos = self.p1_pos
        cloned.p2_pos = self.p2_pos
        
        # Copy directions
        cloned.p1_dir = self.p1_dir
        cloned.p2_dir = self.p2_dir
        
        # Copy game state
        cloned.p1_dead = self.p1_dead
        cloned.p2_dead = self.p2_dead
        cloned.turn = self.turn
        
        return cloned
    
    def _compute_new_position(self, pos, action):
        """
        Compute new position from current position and action.
        
        Args:
            pos: Current position (row, col)
            action: Action index
            
        Returns:
            New position (row, col)
        """
        dr, dc = self.DIRECTIONS[action]
        return (pos[0] + dr, pos[1] + dc)
    
    def _is_crash(self, pos):
        """
        Check if a position would result in a crash.
        
        Args:
            pos: Position to check (row, col)
            
        Returns:
            True if position is invalid (outside board or occupied)
        """
        row, col = pos
        
        # Check bounds
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return True
        
        # Check if cell is occupied
        if self.grid[row, col] != 0:
            return True
        
        return False
    
    def _get_obs(self, who):
        """
        Get observation tensor from perspective of a player.
        
        Observation format (8 channels):
        0: my_trail
        1: opp_trail
        2: free
        3: my_head
        4: opp_head
        5: my_last_dir (constant value across grid)
        6: opp_last_dir (constant value across grid)
        7: boosts (always zeros)
        
        Args:
            who: 'p1' or 'p2'
            
        Returns:
            Observation tensor of shape (8, rows, cols) as float32
        """
        obs = np.zeros((8, self.rows, self.cols), dtype=np.float32)
        
        # Determine which player is "me" and "opponent"
        if who == 'p1':
            my_id = 1
            opp_id = 2
            my_pos = self.p1_pos
            opp_pos = self.p2_pos
            my_dir = self.p1_dir
            opp_dir = self.p2_dir
        else:
            my_id = 2
            opp_id = 1
            my_pos = self.p2_pos
            opp_pos = self.p1_pos
            my_dir = self.p2_dir
            opp_dir = self.p1_dir
        
        # Channel 0: my_trail
        obs[0] = (self.grid == my_id).astype(np.float32)
        
        # Channel 1: opp_trail
        obs[1] = (self.grid == opp_id).astype(np.float32)
        
        # Channel 2: free
        obs[2] = (self.grid == 0).astype(np.float32)
        
        # Heads remain part of the trail (occupied on grid), but we add separate
        # head-only channels (3 and 4) for RL policy clarity. This is intentional.
        
        # Channel 3: my_head
        if my_pos is not None:
            obs[3, my_pos[0], my_pos[1]] = 1.0
        
        # Channel 4: opp_head
        if opp_pos is not None:
            obs[4, opp_pos[0], opp_pos[1]] = 1.0
        
        # Channel 5: my_last_dir (constant raster)
        obs[5].fill(float(my_dir))
        
        # Channel 6: opp_last_dir (constant raster)
        obs[6].fill(float(opp_dir))
        
        # Channel 7: boosts (always zeros)
        # Already initialized to zeros
        
        return obs
