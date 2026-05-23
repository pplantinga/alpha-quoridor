"""Benchmark: batched vs. sequential self-play inference speed."""
import sys
import time

import torch

# Allow imports from src/
sys.path.insert(0, "src")

from model.network import QuoridorNet
from training.self_play import run_self_play_games_batched
from utils.config import Config, MCTSConfig, ModelConfig, TrainingConfig

# ---- Tiny config for speed ----
cfg = Config(
    board_size=5,
    walls_per_player=3,
    model=ModelConfig(hidden_dim=32, num_blocks=1),
    training=TrainingConfig(num_self_play_games=1, batch_size=2, lr=0.001, num_iterations=1, buffer_size=100),
    mcts=MCTSConfig(num_simulations=20, c_puct=1.5, temperature=1.0),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

network = QuoridorNet(cfg.board_size, cfg.model).to(device)
network.eval()

NUM_GAMES = 16

# Warm-up
run_self_play_games_batched(network, cfg, num_games=1, device=device)

# --- Batched: all 16 simultaneously ---
t0 = time.perf_counter()
exps = run_self_play_games_batched(network, cfg, num_games=NUM_GAMES, device=device)
t1 = time.perf_counter()
batched_time = t1 - t0

# --- Sequential: 16 games one by one ---
t0 = time.perf_counter()
for _ in range(NUM_GAMES):
    run_self_play_games_batched(network, cfg, num_games=1, device=device)
t1 = time.perf_counter()
seq_time = t1 - t0

print(f"\n{'='*50}")
print(f"Games:      {NUM_GAMES}")
print(f"Sequential: {seq_time:.2f}s  ({seq_time/NUM_GAMES:.3f}s/game)")
print(f"Batched:    {batched_time:.2f}s  ({batched_time/NUM_GAMES:.3f}s/game)")
print(f"Speedup:    {seq_time/batched_time:.2f}x")
print(f"Experiences: {len(exps)}")
