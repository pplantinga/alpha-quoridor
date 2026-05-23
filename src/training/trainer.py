"""Trainer for the Alpha Quoridor agent."""

import os
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim

from model.network import QuoridorNet
from training.buffer import ExperienceBuffer
from training.self_play import _self_play_worker, run_self_play_games_batched
from utils.config import Config


class Trainer:
    """Handles the self-play and neural network training loop."""

    def __init__(self, network: QuoridorNet, config: Config, device: torch.device):
        self.network: QuoridorNet = network
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.training.lr)
        self.buffer = ExperienceBuffer(max_size=config.training.buffer_size)
        self.scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

        if config.model.use_compile:
            try:
                print("Compiling model with torch.compile...")
                self.network = torch.compile(self.network) # type: ignore
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}. Falling back to default.")

    def train_step(self, batch_size: int) -> dict[str, float]:
        """Perform one gradient descent step.

        Returns:
            Dictionary with loss metrics.
        """
        self.network.train()

        state_batch, policy_batch, value_batch = self.buffer.sample(batch_size)
        state_batch = state_batch.to(self.device)
        policy_batch = policy_batch.to(self.device)
        value_batch = value_batch.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        pred_policy_logits, pred_value = self.network(state_batch)

        # Value loss: Mean squared error
        value_loss = F.mse_loss(pred_value, value_batch)

        # Policy loss: Cross entropy
        # policy_batch represents target probabilities.
        # F.cross_entropy expects class indices or probabilities
        policy_loss = F.cross_entropy(pred_policy_logits, policy_batch)

        # Total loss
        loss = value_loss + policy_loss

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    def update_curriculum(self, epoch: int, schedule: list[list[int]]) -> None:
        """Update training parameters based on the current epoch and schedule.

        Args:
            epoch: Current training epoch.
            schedule: List of [start_epoch, walls_count] pairs.
        """
        # Find the latest applicable wall count
        applicable_walls = self.config.walls_per_player
        for s_epoch, s_walls in sorted(schedule, key=lambda x: x[0]):
            if epoch >= s_epoch:
                applicable_walls = s_walls
            else:
                break

        if self.config.walls_per_player != applicable_walls:
            print(f"Curriculum Update: Epoch {epoch}, setting walls_per_player to {applicable_walls}")
            self.config.walls_per_player = applicable_walls

    def run_iteration(self) -> dict[str, float]:
        """Run one iteration of self-play data generation and training."""
        # 1. Self-Play --- run all games at once, batching GPU inference across trees
        num_games = self.config.training.num_self_play_games
        num_workers = self.config.training.num_workers
        print(f"Generating data via self-play for {num_games} games using {num_workers} workers (Walls: {self.config.walls_per_player})...")

        if num_workers <= 1:
            experiences = run_self_play_games_batched(
                self.network,
                self.config,
                num_games=num_games,
                device=self.device,
            )
        else:
            # Use multiprocessing
            games_per_worker = (num_games + num_workers - 1) // num_workers
            processes = []
            result_queue = mp.Queue()

            # We need to send the model's state_dict to the workers
            # Since torch.compile might be used, we need the underlying module
            orig_network: QuoridorNet = self.network
            if hasattr(self.network, "_orig_mod"):
                orig_network = self.network._orig_mod # type: ignore

            state_dict = {k: v.cpu() for k, v in orig_network.state_dict().items()}

            for i in range(num_workers):
                worker_games = min(games_per_worker, num_games - i * games_per_worker)
                if worker_games <= 0:
                    continue
                p = mp.Process(
                    target=_self_play_worker,
                    args=(state_dict, self.config, worker_games, self.device.type, result_queue)
                )
                p.start()
                processes.append(p)

            experiences = []
            for _ in range(len(processes)):
                raw_exps = result_queue.get()
                # Convert back to torch
                for s_np, p_np, v in raw_exps:
                    experiences.append((torch.from_numpy(s_np), torch.from_numpy(p_np), v))

            for p in processes:
                p.join()

        for exp in experiences:
            self.buffer.add(*exp)

        # 2. Training
        print(f"Training for {self.config.training.num_iterations} batches (batch size {self.config.training.batch_size})...")
        avg_losses = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        # Make sure we have enough data
        b_size = self.config.training.batch_size
        n_iters = self.config.training.num_iterations

        if len(self.buffer) < b_size:
            print(f"Not enough data in buffer ({len(self.buffer)} < {b_size}). Skipping training.")
            return avg_losses

        for _ in range(n_iters):
            losses = self.train_step(b_size)
            for k, v in losses.items():
                avg_losses[k] += v

        for k in avg_losses:
            avg_losses[k] /= n_iters

        return avg_losses

    def save_checkpoint(self, path: Path) -> None:
        """Save the model weights to disk."""
        os.makedirs(path.parent, exist_ok=True)
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model and optimizer state from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Determine the network to load into (handle compiled model)
        target_network: QuoridorNet = self.network
        if hasattr(self.network, "_orig_mod"):
            target_network = self.network._orig_mod # type: ignore

        state_dict = checkpoint["model_state_dict"]
        # Strip _orig_mod. prefix if loading a compiled checkpoint into a non-compiled model or vice-versa
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[10:] if k.startswith("_orig_mod.") else k
            new_state_dict[name] = v

        target_network.load_state_dict(new_state_dict)

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
