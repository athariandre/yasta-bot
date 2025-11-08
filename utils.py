"""
Utility functions for deterministic behavior and logging.
Part of Workflow A - Core Engine Implementation.
"""

import random
import numpy as np


def set_seed(seed):
    """
    Seed Python's random and numpy's RNG for deterministic behavior.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)


class Logger:
    """
    Minimal deterministic logger without external dependencies.
    """
    
    def __init__(self):
        pass
    
    def info(self, msg):
        """Log info message."""
        print(f"[INFO] {msg}")
    
    def debug(self, msg):
        """Log debug message."""
        print(f"[DEBUG] {msg}")
