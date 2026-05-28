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


@dataclass
class RewardConfig:
    """Weights that control the minimax-style reward shaping signal.

    All weights can be tuned via the ``reward:`` section in any YAML config.

    Attributes:
        progress_weight:  Reward per unit decrease in the current player's
                          shortest-path distance (normalized by board size).
                          A 1-step pawn advance on a 9x9 board contributes
                          progress_weight / 9 to the shaped reward.
        block_weight:     Reward per unit *increase* in the opponent's
                          shortest-path distance.  Non-zero for wall placements
                          that don't move our pawn, fixing a gap in the old code.
        draw_penalty:     Base value assigned to every position in a game that
                          ends in a timeout or 3-fold repetition.  Slightly
                          negative to discourage oscillation.
        draw_heuristic_w: Weight of the minimax heuristic evaluation added on
                          top of draw_penalty for draw/timeout positions, so the
                          network still learns a board-position gradient even
                          from drawn games.
    """
    progress_weight: float = 0.3
    block_weight: float = 0.2
    draw_penalty: float = -0.05
    draw_heuristic_w: float = 0.4


def _make_config(data: dict) -> "Config":
    model = ModelConfig(**data.get("model", {}))
    training = TrainingConfig(**data.get("training", {}))
    mcts = MCTSConfig(**data.get("mcts", {}))
    reward = RewardConfig(**data.get("reward", {}))
    return Config(
        board_size=data.get("board_size", 9),
        walls_per_player=data.get("walls_per_player", 10),
        model=model,
        training=training,
        mcts=mcts,
        reward=reward,
    )


@dataclass
class Config:
    board_size: int = 9
    walls_per_player: int = 10
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


def load_config(path: Path) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return _make_config(data)
