"""
Tests for Workflow A modules: env, features, heuristic_bot, utils.

These tests verify:
- Determinism
- Core environment mechanics
- Feature computation correctness
- Heuristic bot functionality
"""

import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, Logger
from env import TronEnv
from features import reachable_area, voronoi_counts, is_tunnel, simulate_step
from heuristic_bot import choose_action


def test_set_seed():
    """Test that set_seed provides deterministic behavior."""
    print("Testing set_seed...")
    
    set_seed(42)
    r1 = np.random.rand()
    
    set_seed(42)
    r2 = np.random.rand()
    
    assert r1 == r2, "set_seed should produce deterministic results"
    print("✓ set_seed works correctly")


def test_logger():
    """Test Logger class."""
    print("\nTesting Logger...")
    
    logger = Logger()
    logger.info("Test info message")
    logger.debug("Test debug message")
    
    print("✓ Logger works correctly")


def test_env_reset():
    """Test environment reset."""
    print("\nTesting TronEnv.reset...")
    
    env = TronEnv(rows=18, cols=20)
    obs = env.reset()
    
    # Check observation shape
    assert obs.shape == (8, 18, 20), f"Expected shape (8, 18, 20), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"
    
    # Check initial positions
    assert env.p1_pos == (9, 5), f"Expected p1 at (9, 5), got {env.p1_pos}"
    assert env.p2_pos == (9, 15), f"Expected p2 at (9, 15), got {env.p2_pos}"
    
    # Check grid
    assert env.grid[env.p1_pos] == 1, "p1 position should be marked on grid"
    assert env.grid[env.p2_pos] == 2, "p2 position should be marked on grid"
    
    # Check game state
    assert not env.p1_dead, "p1 should not be dead at start"
    assert not env.p2_dead, "p2 should not be dead at start"
    assert env.turn == 0, "Turn should be 0 at start"
    
    print("✓ TronEnv.reset works correctly")


def test_env_step():
    """Test environment step function."""
    print("\nTesting TronEnv.step...")
    
    env = TronEnv(rows=18, cols=20)
    env.reset()
    
    # Take one step
    obs, reward_my, reward_opp, done, info = env.step(TronEnv.RIGHT, TronEnv.LEFT)
    
    # Check rewards are 0
    assert reward_my == 0, "Reward should be 0 during game"
    assert reward_opp == 0, "Reward should be 0 during game"
    
    # Check info dict
    assert "p1_dead" in info, "info should contain p1_dead"
    assert "p2_dead" in info, "info should contain p2_dead"
    assert "turn" in info, "info should contain turn"
    assert info["turn"] == 1, "Turn should be 1 after first step"
    
    # Check observation shape
    assert obs.shape == (8, 18, 20), f"Expected shape (8, 18, 20), got {obs.shape}"
    
    print("✓ TronEnv.step works correctly")


def test_env_crash():
    """Test crash detection."""
    print("\nTesting crash detection...")
    
    env = TronEnv(rows=5, cols=5)
    env.reset()
    
    # Move p1 up repeatedly until crash
    for i in range(10):
        obs, reward_my, reward_opp, done, info = env.step(TronEnv.UP, TronEnv.DOWN)
        if done:
            break
    
    assert done, "Game should end when player crashes"
    assert env.p1_dead, "p1 should be dead after crashing into boundary"
    
    print("✓ Crash detection works correctly")


def test_env_head_on_collision():
    """Test head-on collision."""
    print("\nTesting head-on collision...")
    
    env = TronEnv(rows=5, cols=10)
    env.reset()
    
    # Set up scenario where players will collide
    # Move them toward each other
    for i in range(10):
        obs, reward_my, reward_opp, done, info = env.step(TronEnv.RIGHT, TronEnv.LEFT)
        if done:
            break
    
    # At least one should be dead (or both in head-on)
    assert env.p1_dead or env.p2_dead, "At least one player should be dead"
    
    print("✓ Head-on collision works correctly")


def test_env_clone():
    """Test environment cloning."""
    print("\nTesting TronEnv.clone...")
    
    env = TronEnv(rows=18, cols=20)
    env.reset()
    env.step(TronEnv.RIGHT, TronEnv.LEFT)
    
    # Clone environment
    cloned = env.clone()
    
    # Verify deep copy
    assert cloned is not env, "Clone should be a different object"
    assert not np.shares_memory(cloned.grid, env.grid), "Grid should be deep copied"
    
    # Verify state matches
    assert cloned.p1_pos == env.p1_pos, "Positions should match"
    assert cloned.p2_pos == env.p2_pos, "Positions should match"
    assert cloned.turn == env.turn, "Turn should match"
    assert np.array_equal(cloned.grid, env.grid), "Grid should match"
    
    # Verify independence
    cloned.step(TronEnv.UP, TronEnv.DOWN)
    assert cloned.turn != env.turn, "Cloned env should be independent"
    
    print("✓ TronEnv.clone works correctly")


def test_legal_moves():
    """Test legal moves detection."""
    print("\nTesting legal_moves...")
    
    env = TronEnv(rows=5, cols=5)
    env.reset()
    
    legal = env.legal_moves('p1')
    assert isinstance(legal, list), "legal_moves should return a list"
    assert len(legal) > 0, "There should be some legal moves initially"
    
    print("✓ legal_moves works correctly")


def test_reachable_area():
    """Test reachable_area function."""
    print("\nTesting reachable_area...")
    
    # Create simple grid
    grid = np.zeros((5, 5), dtype=np.int8)
    head = (2, 2)
    
    area = reachable_area(grid, head)
    assert area == 25, f"Empty 5x5 grid should have 25 reachable cells, got {area}"
    
    # Add obstacles
    grid[0, :] = 1  # Block top row
    area = reachable_area(grid, head)
    assert area == 20, f"Expected 20 reachable cells, got {area}"
    
    print("✓ reachable_area works correctly")


def test_voronoi_counts():
    """Test voronoi_counts function."""
    print("\nTesting voronoi_counts...")
    
    # Create simple grid
    grid = np.zeros((5, 5), dtype=np.int8)
    head_me = (2, 1)
    head_opp = (2, 3)
    
    me, opp, tie = voronoi_counts(grid, head_me, head_opp)
    
    # Basic sanity checks
    assert me >= 0, "me count should be non-negative"
    assert opp >= 0, "opp count should be non-negative"
    assert tie >= 0, "tie count should be non-negative"
    assert me + opp + tie <= 25, "Total should not exceed grid size"
    
    print(f"  Voronoi: me={me}, opp={opp}, tie={tie}")
    print("✓ voronoi_counts works correctly")


def test_is_tunnel():
    """Test is_tunnel function."""
    print("\nTesting is_tunnel...")
    
    # Create corridor
    grid = np.ones((5, 5), dtype=np.int8)
    grid[2, :] = 0  # Create horizontal corridor
    
    # Middle of corridor should be tunnel
    result = is_tunnel(grid, (2, 2))
    assert result, "Middle of 1-wide corridor should be tunnel"
    
    # Open space
    grid = np.zeros((5, 5), dtype=np.int8)
    result = is_tunnel(grid, (2, 2))
    assert not result, "Open space should not be tunnel"
    
    print("✓ is_tunnel works correctly")


def test_simulate_step():
    """Test simulate_step function."""
    print("\nTesting simulate_step...")
    
    grid = np.zeros((5, 5), dtype=np.int8)
    head = (2, 2)
    
    # Simulate move up
    new_head, crashed = simulate_step(grid, head, 0)
    assert new_head == (1, 2), f"Expected (1, 2), got {new_head}"
    assert not crashed, "Should not crash in open space"
    
    # Verify grid is not modified
    assert np.all(grid == 0), "simulate_step should not modify input grid"
    
    # Simulate crash into boundary
    head = (0, 0)
    new_head, crashed = simulate_step(grid, head, 0)  # Up from top-left
    assert crashed, "Should crash when moving out of bounds"
    
    print("✓ simulate_step works correctly")


def test_heuristic_bot():
    """Test heuristic bot."""
    print("\nTesting heuristic_bot...")
    
    env = TronEnv(rows=18, cols=20)
    env.reset()
    
    # Test action selection
    action = choose_action(env, 'p1')
    assert action in [0, 1, 2, 3], f"Action should be in range [0, 3], got {action}"
    
    # Test determinism
    env2 = TronEnv(rows=18, cols=20)
    env2.reset()
    action2 = choose_action(env2, 'p1')
    assert action == action2, "Heuristic bot should be deterministic"
    
    print("✓ heuristic_bot works correctly")


def test_determinism():
    """Test overall determinism of the system."""
    print("\nTesting determinism...")
    
    # Run two identical games
    set_seed(42)
    env1 = TronEnv(rows=10, cols=10)
    env1.reset()
    
    set_seed(42)
    env2 = TronEnv(rows=10, cols=10)
    env2.reset()
    
    # Run 10 steps with heuristic bot
    for i in range(10):
        action1 = choose_action(env1, 'p1')
        action2 = choose_action(env2, 'p1')
        assert action1 == action2, f"Actions should match at step {i}"
        
        # For opponent, use same action
        _, _, _, done1, _ = env1.step(action1, action1)
        _, _, _, done2, _ = env2.step(action2, action2)
        
        if done1 or done2:
            break
        
        assert np.array_equal(env1.grid, env2.grid), f"Grids should match at step {i}"
    
    print("✓ System is deterministic")


def test_observation_format():
    """Test that observation tensor has correct format."""
    print("\nTesting observation format...")
    
    env = TronEnv(rows=18, cols=20)
    obs = env.reset()
    
    # Verify shape
    assert obs.shape == (8, 18, 20), f"Expected shape (8, 18, 20), got {obs.shape}"
    
    # Verify channels
    # Channel 0: my_trail
    assert np.sum(obs[0]) > 0, "my_trail should have at least one cell"
    
    # Channel 1: opp_trail
    assert np.sum(obs[1]) > 0, "opp_trail should have at least one cell"
    
    # Channel 2: free
    assert np.sum(obs[2]) > 0, "free should have many cells"
    
    # Channel 3: my_head
    assert np.sum(obs[3]) == 1, "my_head should have exactly one cell"
    
    # Channel 4: opp_head
    assert np.sum(obs[4]) == 1, "opp_head should have exactly one cell"
    
    # Channel 5: my_last_dir (constant raster)
    assert np.all(obs[5] == obs[5, 0, 0]), "my_last_dir should be constant"
    
    # Channel 6: opp_last_dir (constant raster)
    assert np.all(obs[6] == obs[6, 0, 0]), "opp_last_dir should be constant"
    
    # Channel 7: boosts (always zeros)
    assert np.all(obs[7] == 0), "boosts should be all zeros"
    
    print("✓ Observation format is correct")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Workflow A Tests")
    print("=" * 60)
    
    test_set_seed()
    test_logger()
    test_env_reset()
    test_env_step()
    test_env_crash()
    test_env_head_on_collision()
    test_env_clone()
    test_legal_moves()
    test_reachable_area()
    test_voronoi_counts()
    test_is_tunnel()
    test_simulate_step()
    test_heuristic_bot()
    test_determinism()
    test_observation_format()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
