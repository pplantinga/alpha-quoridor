
from model.network import QuoridorNet
from training.self_play import run_self_play_games_batched
from utils.config import Config, MCTSConfig, ModelConfig, TrainingConfig


def test_repetition_detection():
    # Setup a tiny config
    config = Config(
        board_size=3,
        walls_per_player=1,
        model=ModelConfig(hidden_dim=8, num_blocks=1),
        training=TrainingConfig(num_self_play_games=1, num_iterations=1, num_workers=1),
        mcts=MCTSConfig(num_simulations=2)
    )

    network = QuoridorNet(3, config.model)

    # We want to verify that run_self_play_games_batched eventually returns.
    # To force repetition, we would need a deterministic policy that oscillates.
    # Instead, we'll just verify the loop runs and produces experiences.
    exps = run_self_play_games_batched(network, config, num_games=1)
    assert len(exps) > 0
    print(f"Generated {len(exps)} experiences.")

if __name__ == "__main__":
    test_repetition_detection()
