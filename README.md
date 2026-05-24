# Alpha Quoridor

An AlphaGo-style AI player for the board game [Quoridor](https://en.wikipedia.org/wiki/Quoridor), built with PyTorch.

## Overview

This project implements a deep reinforcement learning agent that learns to play Quoridor through self-play, using:

- **Neural Network**: Combined policy and value network
- **Monte Carlo Tree Search (MCTS)**: With PUCT selection for move planning
- **Self-Play Training**: Iterative improvement through generated game data

## Features

- Configurable board sizes (3x3, 5x5, 7x7, 9x9)
- Configurable wall counts per player
- Single-machine training pipeline
- CLI for playing against the trained AI

## Project Structure

```
alpha-quoridor/
├── src/
│   ├── game/          # Game engine (board, rules, validation)
│   ├── model/         # Neural network (policy + value heads)
│   ├── mcts/          # Monte Carlo Tree Search
│   ├── training/      # Self-play, experience buffer, training loop
│   ├── agent/         # AI player interface
│   └── utils/         # Config, logging, pathfinding
├── configs/           # YAML configuration files
├── scripts/           # Training and evaluation scripts
├── tests/             # Unit and integration tests
└── checkpoints/       # Saved model weights (gitignored)
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0

## Setup

This project uses `uv` for dependency management.

```bash
# Install dependencies and create virtual environment
uv sync --all-extras
```

## Usage

### Training

```bash
# Start self-play training with default config
uv run python scripts/train.py

# Use a custom config
uv run python scripts/train.py --config configs/small.yaml
```

### Evaluation

```bash
# Evaluate a trained model against a random agent
uv run python scripts/eval.py --checkpoint checkpoints/best.pt
```

### Play Against the AI

```bash
# Launch CLI game
uv run python scripts/play.py --checkpoint checkpoints/best.pt
```

## Configuration

Board size and wall count are controlled via config files:

```yaml
# configs/default.yaml
board_size: 9
walls_per_player: 10
model:
  hidden_dim: 256
  num_blocks: 4
training:
  num_self_play_games: 1000
  batch_size: 256
  lr: 0.001
  num_iterations: 100
```

Smaller boards for fast iteration:

```yaml
# configs/small.yaml
board_size: 5
walls_per_player: 3
```

## License

MIT
