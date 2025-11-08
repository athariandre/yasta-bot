"""
Example usage of Workflow A modules.
Demonstrates a simple game between two heuristic bots.
"""

from env import TronEnv
from heuristic_bot import choose_action
from utils import set_seed, Logger

def run_example_game():
    """Run an example game with heuristic bots."""
    logger = Logger()
    
    # Set seed for deterministic behavior
    set_seed(42)
    
    # Create environment
    logger.info("Creating Tron environment (18x20)")
    env = TronEnv(rows=18, cols=20)
    
    # Reset environment
    logger.info("Resetting environment")
    obs = env.reset()
    logger.info(f"Initial observation shape: {obs.shape}")
    logger.info(f"P1 starting at: {env.p1_pos}")
    logger.info(f"P2 starting at: {env.p2_pos}")
    
    # Play game
    logger.info("\nPlaying game with heuristic bots...")
    max_turns = 100
    
    for turn in range(max_turns):
        # Get actions from heuristic bots
        action_p1 = choose_action(env, 'p1')
        action_p2 = choose_action(env, 'p2')
        
        # Take step
        obs, reward_p1, reward_p2, done, info = env.step(action_p1, action_p2)
        
        if turn % 10 == 0:
            logger.info(f"Turn {turn}: P1 action={action_p1}, P2 action={action_p2}")
        
        if done:
            logger.info(f"\nGame ended at turn {info['turn']}")
            logger.info(f"P1 dead: {info['p1_dead']}")
            logger.info(f"P2 dead: {info['p2_dead']}")
            
            if info['p1_dead'] and info['p2_dead']:
                logger.info("Result: DRAW (both crashed)")
            elif info['p1_dead']:
                logger.info("Result: P2 WINS")
            else:
                logger.info("Result: P1 WINS")
            break
    else:
        logger.info(f"\nGame reached maximum turns ({max_turns})")
    
    # Display final grid
    logger.info("\nFinal grid:")
    logger.info("0 = empty, 1 = P1 trail, 2 = P2 trail")
    print(env.grid)


if __name__ == "__main__":
    run_example_game()
