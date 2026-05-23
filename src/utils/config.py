"""Configuration dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_blocks: int = 4
    use_compile: bool = True


@dataclass
class TrainingConfig:
    num_self_play_games: int = 1000
    batch_size: int = 256
    lr: float = 0.001
    num_iterations: int = 100
    buffer_size: int = 50000
    num_workers: int = 4


@dataclass
class MCTSConfig:
    num_simulations: int = 800
    c_puct: float = 1.5
    temperature: float = 1.0
    dirichlet_noise_alpha: float = 0.3
    dirichlet_noise_epsilon: float = 0.25


def _make_config(data: dict) -> "Config":
    model = ModelConfig(**data.get("model", {}))
    training = TrainingConfig(**data.get("training", {}))
    mcts = MCTSConfig(**data.get("mcts", {}))
    return Config(
        board_size=data.get("board_size", 9),
        walls_per_player=data.get("walls_per_player", 10),
        model=model,
        training=training,
        mcts=mcts,
    )


@dataclass
class Config:
    board_size: int = 9
    walls_per_player: int = 10
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)


def load_config(path: Path) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return _make_config(data)
