"""
Residual CNN Policy Network (AlphaZero-style).
Part of Workflow B - RL Training Pipeline.

Architecture:
- Input: (8, 18, 20) float tensor
- Conv Stem + 10 Residual Blocks
- Dual heads: Policy (4 actions) and Value (scalar)
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    Architecture:
        Conv2d(64 → 64, kernel=3, padding=1)
        BatchNorm2d(64)
        ReLU
        Conv2d(64 → 64, kernel=3, padding=1)
        BatchNorm2d(64)
        Skip connection
        ReLU
    """
    
    def __init__(self, channels: int = 64):
        """
        Initialize residual block.
        
        Args:
            channels: Number of channels (default 64)
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Output tensor with same shape as input
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        out = out + identity
        out = self.relu2(out)
        
        return out


class TronPolicy(nn.Module):
    """
    Residual CNN policy network for Tron.
    
    Architecture:
        Conv Stem: (8 → 64 channels)
        Residual Tower: 10 residual blocks
        Policy Head: 4 action logits
        Value Head: scalar value in [-1, 1]
    """
    
    def __init__(self):
        """Initialize the policy network."""
        super(TronPolicy, self).__init__()
        
        # Conv Stem
        self.conv_stem = nn.Conv2d(8, 64, kernel_size=3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(64)
        self.relu_stem = nn.ReLU()
        
        # Residual Tower: 10 blocks
        self.residual_tower = nn.ModuleList([
            ResidualBlock(64) for _ in range(10)
        ])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(64, 4, kernel_size=1, bias=False)
        self.policy_flatten = nn.Flatten()
        
        # Value Head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.value_relu = nn.ReLU()
        self.value_flatten = nn.Flatten()
        self.value_fc1 = nn.Linear(18 * 20, 64)
        self.value_fc1_relu = nn.ReLU()
        self.value_fc2 = nn.Linear(64, 1)
        self.value_tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 8, 18, 20)
            
        Returns:
            Tuple of (policy_logits, value_scalar):
                - policy_logits: (batch, 4) action logits
                - value_scalar: (batch, 1) value estimate in [-1, 1]
        """
        # Conv Stem
        x = self.conv_stem(x)
        x = self.bn_stem(x)
        x = self.relu_stem(x)
        
        # Residual Tower
        for residual_block in self.residual_tower:
            x = residual_block(x)
        
        # Policy Head
        policy = self.policy_conv(x)  # (batch, 4, 18, 20)
        policy = self.policy_flatten(policy)  # (batch, 4*18*20)
        # Sum over spatial dimensions to get (batch, 4)
        policy = policy.view(policy.size(0), 4, -1).sum(dim=2)
        
        # Value Head
        value = self.value_conv(x)  # (batch, 1, 18, 20)
        value = self.value_relu(value)
        value = self.value_flatten(value)  # (batch, 18*20)
        value = self.value_fc1(value)
        value = self.value_fc1_relu(value)
        value = self.value_fc2(value)  # (batch, 1)
        value = self.value_tanh(value)
        
        return policy, value
