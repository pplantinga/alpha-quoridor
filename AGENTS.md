# AGENTS.md - Developer Guide

## Project Overview

Alpha Quoridor is an AlphaGo-style AI for the board game Quoridor. The system uses self-play with Monte Carlo Tree Search and a neural network with policy and value heads.

## Key Architecture Decisions

- **Framework**: PyTorch
- **Players**: 2-player only
- **Board sizes**: Odd squares (3x3, 5x5, 7x7, 9x9), configurable
- **Walls**: Configurable count per player (scales with board size)
- **Training**: Single-machine, iterative self-play
- **Performance**: High-performance Cython-based engine for move validation and pathfinding.
- **Config**: YAML files in `configs/`

## Directory Structure

```
src/
  game/       - Board state, move generation, validation (including fast Cython engine)
  model/      - PyTorch neural network (shared trunk + policy/value heads)
  mcts/       - MCTS tree, PUCT selection, parallel self-play
  training/   - Self-play loop, experience buffer, loss computation, optimizer
  agent/      - MCTS-based player, random baseline, human CLI player
  utils/      - Config loading, logging, BFS pathfinding, helpers
```

## Running Code

### Environment

This project uses `uv` for dependency management.

```bash
# Install dependencies and create virtual environment
uv sync --all-extras

# Build Cython extensions for the fast game engine
uv run python setup.py build_ext --inplace
```

### Commands

```bash
# Train
uv run python scripts/train.py --config configs/default.yaml

# Evaluate
uv run python scripts/eval.py --checkpoint <path>

# Play interactively
uv run python scripts/play.py --checkpoint <path>

# Run tests
uv run pytest tests/

# Lint
uv run ruff check src/ tests/

# Type check
uv run ty check src/
```

### Configuration

All hyperparameters live in `configs/`. Key parameters:

- `board_size`: Grid dimension (must be odd, >= 3)
- `walls_per_player`: Wall count per player
- `model.hidden_dim`: Network hidden dimension
- `model.num_blocks`: Residual block count
- `training.num_self_play_games`: Games per iteration
- `training.batch_size`: Training batch size
- `training.lr`: Learning rate
- `training.num_iterations`: Total training iterations
- `mcts.num_simulations`: MCTS simulations per move
- `mcts.c_puct`: Exploration constant

## Code Conventions

- Type hints on all function signatures
- Dataclasses for configuration objects
- PyTorch nn.Module for all network components
- No comments unless clarifying non-obvious logic
- Follow existing patterns in neighboring files
- Run `ruff check` and `ty check` before committing

## Testing

- Unit tests in `tests/` mirror `src/` structure
- Test game logic thoroughly (path validation, wall blocking, shortest path checks)
- Use `pytest` as the test framework
