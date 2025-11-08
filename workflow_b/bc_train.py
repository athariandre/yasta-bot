"""
Behavior Cloning Training Script.
Part of Workflow B - RL Training Pipeline.

Trains the policy network on heuristic bot demonstrations.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow_b.policy import TronPolicy


class BCDataset(Dataset):
    """PyTorch dataset wrapper for behavior cloning data."""
    
    def __init__(self, obs: np.ndarray, actions: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            obs: Observations array of shape (N, 8, 18, 20)
            actions: Actions array of shape (N,)
        """
        self.obs = torch.from_numpy(obs).float()
        self.actions = torch.from_numpy(actions).long()
    
    def __len__(self) -> int:
        return len(self.obs)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.obs[idx], self.actions[idx]


def train_bc(dataset_path: str, output_model_path: str, 
             epochs: int = 30, batch_size: int = 128, lr: float = 1e-3) -> None:
    """
    Train behavior cloning model.
    
    Args:
        dataset_path: Path to .npz dataset file
        output_model_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    """
    print("Loading dataset...")
    data = np.load(dataset_path)
    obs = data['obs']
    actions = data['actions']
    
    print(f"Dataset size: {len(obs)} samples")
    print(f"Observation shape: {obs.shape}")
    print(f"Actions shape: {actions.shape}")
    
    # Create dataset and dataloader
    dataset = BCDataset(obs, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    print("\nInitializing model...")
    model = TronPolicy()
    model.train()
    
    # Loss and optimizer (only train on policy head)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    loss_history = []
    acc_history = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_obs, batch_actions in dataloader:
            # Forward pass
            policy_logits, _ = model(batch_obs)
            
            # Compute loss (only on policy head)
            loss = criterion(policy_logits, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Compute accuracy
            predictions = torch.argmax(policy_logits, dim=1)
            correct += (predictions == batch_actions).sum().item()
            total += batch_actions.size(0)
        
        # Epoch metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        loss_history.append(avg_loss)
        acc_history.append(accuracy)
        
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Save model
    print(f"\nSaving model to {output_model_path}...")
    torch.save(model.state_dict(), output_model_path)
    
    # Save training curve plot
    plot_path = os.path.join(os.path.dirname(output_model_path), 'bc_loss_curve.png')
    print(f"Saving training curve to {plot_path}...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(loss_history)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('BC Training Loss')
    ax1.grid(True)
    
    # Accuracy curve
    ax2.plot(acc_history)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('BC Training Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print("Training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BC model")
    parser.add_argument("--dataset", type=str, default="models/bc_dataset.npz",
                        help="Path to BC dataset (default: models/bc_dataset.npz)")
    parser.add_argument("--output", type=str, default="models/policy_bc.pth",
                        help="Output model path (default: models/policy_bc.pth)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    train_bc(args.dataset, args.output, args.epochs, args.batch_size, args.lr)
