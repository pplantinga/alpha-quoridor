"""Tests for utils/config.py."""

from pathlib import Path

from utils.config import Config, MCTSConfig, ModelConfig, TrainingConfig, load_config

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


def test_load_small_config() -> None:
    cfg = load_config(CONFIGS_DIR / "small.yaml")
    assert cfg.board_size == 5
    assert cfg.walls_per_player == 3
    assert cfg.model.hidden_dim == 64
    assert cfg.model.num_blocks == 2
    assert cfg.training.num_self_play_games == 50
    assert cfg.mcts.num_simulations == 50


def test_config_defaults() -> None:
    cfg = Config()
    assert cfg.board_size == 9
    assert cfg.walls_per_player == 10
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.mcts, MCTSConfig)


def test_load_config_missing_sections(tmp_path: Path) -> None:
    """Partial YAML should fall back to dataclass defaults."""
    yaml_file = tmp_path / "partial.yaml"
    yaml_file.write_text("board_size: 3\nwalls_per_player: 1\n")
    cfg = load_config(yaml_file)
    assert cfg.board_size == 3
    assert cfg.walls_per_player == 1
    assert cfg.model.hidden_dim == 256  # default
    assert cfg.mcts.num_simulations == 800  # default
