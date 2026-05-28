import torch

from model.network import QuoridorNet
from training.bootstrap_utils import generate_heuristic_data
from training.trainer import Trainer
from utils.config import Config, ModelConfig, TrainingConfig


def test_curriculum_logic():
    config = Config(
        board_size=3,
        walls_per_player=10,
        model=ModelConfig(hidden_dim=8, num_blocks=1)
    )
    network = QuoridorNet(3, config.model)
    trainer = Trainer(network, config, torch.device("cpu"))

    schedule = [[1, 0], [10, 2], [20, 5]]

    # Test epoch 1
    trainer.update_curriculum(1, schedule)
    assert config.walls_per_player == 0

    # Test epoch 5
    trainer.update_curriculum(5, schedule)
    assert config.walls_per_player == 0

    # Test epoch 10
    trainer.update_curriculum(10, schedule)
    assert config.walls_per_player == 2

    # Test epoch 15
    trainer.update_curriculum(15, schedule)
    assert config.walls_per_player == 2

    # Test epoch 25
    trainer.update_curriculum(25, schedule)
    assert config.walls_per_player == 5

def test_minimax_data_generation():
    config = Config(
        board_size=3,
        walls_per_player=2,
        training=TrainingConfig(batch_size=4)
    )

    # Generate data for 2 games
    exps = generate_heuristic_data(config, num_games=2, depth=1)

    assert len(exps) > 0
    s, p, v = exps[0]

    # Check shapes
    assert s.shape == (9, 3, 3)
    assert p.shape == (3 * 3 * 3,)
    assert isinstance(v, float)
    # Values are now continuous in [-1, 1] due to minimax-style reward shaping
    assert -1.0 <= v <= 1.0, f"Bootstrap value out of range: {v}"

if __name__ == "__main__":
    test_curriculum_logic()
    test_minimax_data_generation()
    print("Tests passed!")
