"""
Evaluation Script.
Part of Workflow B - RL Training Pipeline.

Evaluates trained models against different opponents.
"""

import sys
import os
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import TronEnv
from workflow_b.policy import TronPolicy
from heuristic_bot import choose_action


class RandomBot:
    """Random action bot."""
    
    @staticmethod
    def choose_action(env, me):
        legal = env.legal_moves(me)
        if not legal:
            return 0
        return np.random.choice(legal)


class PolicyBot:
    """Bot using trained policy."""
    
    def __init__(self, policy: TronPolicy, deterministic: bool = True):
        self.policy = policy
        self.policy.eval()
        self.deterministic = deterministic
    
    def choose_action(self, env, me):
        obs = env._get_obs(me)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        
        with torch.no_grad():
            policy_logits, _ = self.policy(obs_tensor)
            
            if self.deterministic:
                # Argmax
                action = torch.argmax(policy_logits, dim=1).item()
            else:
                # Sample
                probs = torch.softmax(policy_logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
        
        return action


def play_game(bot1, bot2, seed: int = None) -> tuple[int, int, int]:
    """
    Play a single game between two bots.
    
    Args:
        bot1: Bot playing as p1
        bot2: Bot playing as p2
        seed: Random seed
        
    Returns:
        Tuple of (p1_won, p2_won, draw) where one value is 1 and others are 0
    """
    env = TronEnv(seed=seed)
    env.reset()
    
    done = False
    while not done:
        action_p1 = bot1.choose_action(env, 'p1')
        action_p2 = bot2.choose_action(env, 'p2')
        
        _, _, _, done, info = env.step(action_p1, action_p2)
    
    # Determine outcome
    if info['p1_dead'] and info['p2_dead']:
        return (0, 0, 1)  # Draw
    elif info['p1_dead']:
        return (0, 1, 0)  # P2 won
    else:
        return (1, 0, 0)  # P1 won


def evaluate_model(model_path: str, num_games: int = 100) -> None:
    """
    Evaluate a trained model against various opponents.
    
    Args:
        model_path: Path to model file
        num_games: Number of games to play per matchup
    """
    print(f"Loading model from {model_path}...")
    policy = TronPolicy()
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    
    policy_bot = PolicyBot(policy, deterministic=True)
    
    # Test against different opponents
    opponents = {
        'Heuristic Bot': lambda env, me: choose_action(env, me),
        'Random Bot': RandomBot.choose_action,
        'Self (Mirror)': PolicyBot(policy, deterministic=False).choose_action
    }
    
    print(f"\nEvaluating over {num_games} games per matchup...\n")
    
    for opp_name, opp_bot in opponents.items():
        print(f"VS {opp_name}:")
        
        # Play as P1
        p1_wins = 0
        p2_wins = 0
        draws = 0
        
        for i in range(num_games):
            # Alternate who plays as P1
            if i % 2 == 0:
                # Policy as P1
                w1, w2, d = play_game(policy_bot, 
                                     type('Bot', (), {'choose_action': lambda s, e, m: opp_bot(e, m)})(),
                                     seed=i)
            else:
                # Policy as P2
                w1, w2, d = play_game(type('Bot', (), {'choose_action': lambda s, e, m: opp_bot(e, m)})(),
                                     policy_bot,
                                     seed=i)
                # Swap results since policy is P2
                w1, w2 = w2, w1
            
            p1_wins += w1
            p2_wins += w2
            draws += d
        
        total = p1_wins + p2_wins + draws
        win_rate = 100.0 * p1_wins / total if total > 0 else 0.0
        loss_rate = 100.0 * p2_wins / total if total > 0 else 0.0
        draw_rate = 100.0 * draws / total if total > 0 else 0.0
        
        print(f"  Wins: {p1_wins}/{total} ({win_rate:.1f}%)")
        print(f"  Losses: {p2_wins}/{total} ({loss_rate:.1f}%)")
        print(f"  Draws: {draws}/{total} ({draw_rate:.1f}%)")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, default="models/policy_ppo_latest.pth",
                        help="Path to model file (default: models/policy_ppo_latest.pth)")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games per matchup (default: 100)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    evaluate_model(args.model, args.games)
