"""Training script for Alpha Quoridor."""

import argparse
from pathlib import Path

import torch

from model.network import QuoridorNet
from training.trainer import Trainer
from utils.config import load_config


def main() -> None:
    # Use 'spawn' for CUDA compatibility in multiprocessing
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Train Alpha Quoridor")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    device = torch.device(args.device)

    print(f"Training on {device} with config: {config_path}")
    print(f"Board shape: {config.board_size}x{config.board_size}, Walls: {config.walls_per_player}")

    network = QuoridorNet(config.board_size, config.model).to(device)

    checkpoint_path = Path(args.checkpoint)
    trainer = Trainer(network, config, device)

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        trainer.load_checkpoint(checkpoint_path)

    print("Starting training loop...")
    try:
        # We don't have an outer loop over iterations here,
        # run_iteration itself loops over training.num_iterations batches
        # We'll put a small outer loop just to save intermediate checkpoints.
        for epoch in range(1, 101):
            print(f"\n--- Epoch {epoch} ---")
            losses = trainer.run_iteration()
            print(f"Losses: Total={losses['loss']:.4f}, Policy={losses['policy_loss']:.4f}, Value={losses['value_loss']:.4f}")

            trainer.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}.")

    except KeyboardInterrupt:
        print("\\nTraining interrupted by user. Saving final checkpoint...")
        trainer.save_checkpoint(checkpoint_path)

if __name__ == "__main__":
    main()
