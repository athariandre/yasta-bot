"""
Master Training Script.
Part of Workflow B - RL Training Pipeline.

Orchestrates the complete training pipeline:
1. Generate BC dataset
2. Train BC model
3. Train PPO self-play
"""

import sys
import os
import subprocess
import signal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(cmd: list[str], description: str) -> int:
    """
    Run a command and print its output.
    
    Args:
        cmd: Command to run as list of strings
        description: Description of the command
        
    Returns:
        Return code
    """
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with return code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 130


def main():
    """Run the complete training pipeline."""
    print("=" * 60)
    print("WORKFLOW B - COMPLETE RL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Generate BC dataset
    print("\n[Step 1/3] Generating Behavior Cloning dataset...")
    ret = run_command(
        [sys.executable, "workflow_b/bc_dataset.py", "--num_games", "200"],
        "Generating BC dataset from heuristic bot self-play"
    )
    if ret != 0:
        print("Failed to generate BC dataset")
        return ret
    
    # Step 2: Train BC model
    print("\n[Step 2/3] Training Behavior Cloning model...")
    ret = run_command(
        [sys.executable, "workflow_b/bc_train.py", "--epochs", "30"],
        "Training BC model on heuristic demonstrations"
    )
    if ret != 0:
        print("Failed to train BC model")
        return ret
    
    # Step 3: Train PPO self-play
    print("\n[Step 3/3] Training PPO with self-play...")
    print("\nNote: This will run for a long time. Press Ctrl-C to stop.")
    print("The model will be saved periodically to models/policy_ppo_latest.pth\n")
    
    ret = run_command(
        [sys.executable, "workflow_b/ppo_selfplay.py", "--iterations", "100"],
        "Training PPO with self-play"
    )
    if ret == 130:  # Ctrl-C
        print("\n\nPPO training interrupted by user")
        print("Latest model saved to models/policy_ppo_latest.pth")
        return 0
    elif ret != 0:
        print("Failed to train PPO")
        return ret
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nModels saved:")
    print("  - models/policy_bc.pth (behavior cloning)")
    print("  - models/policy_ppo_latest.pth (PPO self-play)")
    print("\nYou can now:")
    print("  - Evaluate models: python workflow_b/eval.py")
    print("  - Export agent: python workflow_b/make_agent_export.py")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Models have been saved.")
        sys.exit(0)
