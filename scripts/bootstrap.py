"""Bootstrap training script for Alpha Quoridor.

This script implements:
1. Minimax Priming: Generates initial data using a heuristic agent.
2. Wall Curriculum: Gradually increases the number of walls during training.
"""

import argparse
from pathlib import Path

import torch
import yaml

from model.network import QuoridorNet
from training.bootstrap_utils import generate_heuristic_data
from training.trainer import Trainer
from utils.config import load_config


def main() -> None:
    # Use 'spawn' for CUDA compatibility in multiprocessing
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Bootstrap Train Alpha Quoridor")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--bootstrap-config", type=str, default="configs/bootstrap.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bootstrap.pt")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    device = torch.device(args.device)

    # Load bootstrap config
    bootstrap_config_path = Path(args.bootstrap_config)
    if bootstrap_config_path.exists():
        with open(bootstrap_config_path) as f:
            b_config = yaml.safe_load(f) or {}
    else:
        print(f"Warning: Bootstrap config {bootstrap_config_path} not found. Using defaults.")
        b_config = {
            "minimax_priming_games": 50,
            "minimax_depth": 2,
            "wall_curriculum": [[1, 0], [10, 2], [20, 5], [40, 10]],
            "epochs": 100
        }

    print(f"Bootstrapping on {device}")

    network = QuoridorNet(config.board_size, config.model).to(device)
    trainer = Trainer(network, config, device)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        trainer.load_checkpoint(checkpoint_path)

    # --- Phase 1: Minimax Priming ---
    num_priming = b_config.get("minimax_priming_games", 0)
    if num_priming > 0 and len(trainer.buffer) == 0:
        print(f"Phase 1: Generating {num_priming} Minimax priming games...")
        priming_exps = generate_heuristic_data(
            config,
            num_games=num_priming,
            depth=b_config.get("minimax_depth", 2)
        )
        for exp in priming_exps:
            trainer.buffer.add(*exp)
        print(f"Primed buffer with {len(trainer.buffer)} experiences.")

        # Initial training on priming data
        print("Initial training on priming data...")
        trainer.train_step(config.training.batch_size)

    # --- Phase 2: Curriculum Training ---
    print("Phase 2: Starting curriculum training loop...")
    schedule = b_config.get("wall_curriculum", [])
    epochs = b_config.get("epochs", 100)

    try:
        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch} ---")

            # Apply curriculum
            trainer.update_curriculum(epoch, schedule)

            losses = trainer.run_iteration()
            print(f"Losses: Total={losses['loss']:.4f}, Policy={losses['policy_loss']:.4f}, Value={losses['value_loss']:.4f}")

            trainer.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint...")
        trainer.save_checkpoint(checkpoint_path)

if __name__ == "__main__":
    main()
