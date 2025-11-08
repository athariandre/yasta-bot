"""
Agent Exporter for Competition.
Part of Workflow B - RL Training Pipeline.

Creates a standalone agent.py file for tournament submission.
"""

import sys
import os
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow_b.policy import TronPolicy


def create_agent_file(model_path: str, output_path: str = "agent.py") -> None:
    """
    Create standalone agent.py file for competition.
    
    Args:
        model_path: Path to trained model
        output_path: Output path for agent.py
    """
    print(f"Loading model from {model_path}...")
    
    # Load model to get state dict
    policy = TronPolicy()
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    
    # Save state dict to a temporary buffer
    state_dict = policy.state_dict()
    
    # Convert state dict to string representation
    print("Creating agent.py...")
    
    agent_code = '''"""
Competition Agent - Auto-generated from trained model.
Minimal inference-only implementation.
"""

import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu2(out)
        return out


class TronPolicy(nn.Module):
    """Residual CNN policy network."""
    
    def __init__(self):
        super(TronPolicy, self).__init__()
        
        # Conv Stem
        self.conv_stem = nn.Conv2d(8, 64, kernel_size=3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(64)
        self.relu_stem = nn.ReLU()
        
        # Residual Tower
        self.residual_tower = nn.ModuleList([ResidualBlock(64) for _ in range(10)])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(64, 4, kernel_size=1, bias=False)
        self.policy_flatten = nn.Flatten()
        
        # Value Head (not used for action selection, but kept for compatibility)
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.value_relu = nn.ReLU()
        self.value_flatten = nn.Flatten()
        self.value_fc1 = nn.Linear(18 * 20, 64)
        self.value_fc1_relu = nn.ReLU()
        self.value_fc2 = nn.Linear(64, 1)
        self.value_tanh = nn.Tanh()
    
    def forward(self, x):
        # Conv Stem
        x = self.conv_stem(x)
        x = self.bn_stem(x)
        x = self.relu_stem(x)
        
        # Residual Tower
        for residual_block in self.residual_tower:
            x = residual_block(x)
        
        # Policy Head
        policy = self.policy_conv(x)
        policy = self.policy_flatten(policy)
        policy = policy.view(policy.size(0), 4, -1).sum(dim=2)
        
        # Value Head
        value = self.value_conv(x)
        value = self.value_relu(value)
        value = self.value_flatten(value)
        value = self.value_fc1(value)
        value = self.value_fc1_relu(value)
        value = self.value_fc2(value)
        value = self.value_tanh(value)
        
        return policy, value


# Global model instance
_model = None
_initialized = False


def agent(observation):
    """
    Competition agent function.
    
    Args:
        observation: Dictionary containing game state
        
    Returns:
        Action index (0=Up, 1=Right, 2=Down, 3=Left)
    """
    global _model, _initialized
    
    # Initialize model on first call
    if not _initialized:
        _model = TronPolicy()
        _model.load_state_dict(torch.load('__MODEL_WEIGHTS_FILE__'))
        _model.eval()
        _initialized = True
    
    # Extract observation tensor
    # Assuming observation is a dict with 'obs' key containing (8, 18, 20) array
    if isinstance(observation, dict):
        obs = observation.get('obs', observation)
    else:
        obs = observation
    
    # Convert to tensor
    obs_tensor = torch.from_numpy(np.array(obs)).unsqueeze(0).float()
    
    # Get action
    with torch.no_grad():
        policy_logits, _ = _model(obs_tensor)
        action = torch.argmax(policy_logits, dim=1).item()
    
    return action
'''
    
    # Save state dict to a file
    weights_file = output_path.replace('.py', '_weights.pth')
    torch.save(state_dict, weights_file)
    
    # Update agent code with weights file path
    agent_code = agent_code.replace('__MODEL_WEIGHTS_FILE__', weights_file)
    
    # Write agent file
    with open(output_path, 'w') as f:
        f.write(agent_code)
    
    print(f"Agent file created: {output_path}")
    print(f"Model weights saved to: {weights_file}")
    print("\nTo use this agent:")
    print("1. Copy both agent.py and the weights file to your submission")
    print("2. Ensure the weights file path is accessible")
    print("3. Call agent(observation) to get actions")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export competition agent")
    parser.add_argument("--model", type=str, default="models/policy_ppo_latest.pth",
                        help="Path to trained model (default: models/policy_ppo_latest.pth)")
    parser.add_argument("--output", type=str, default="agent.py",
                        help="Output path for agent.py (default: agent.py)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    create_agent_file(args.model, args.output)
