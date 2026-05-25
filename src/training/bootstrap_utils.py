"""Utilities for bootstrapping Alpha Quoridor training."""

import torch

from agent.minimax_agent import MinimaxAgent
from game.board import initial_state
from game.rules import apply_move
from model.network import encode_state, move_to_index
from training.buffer import Experience
from utils.config import Config


def generate_heuristic_data(
    config: Config,
    num_games: int,
    depth: int = 2,
    randomize: bool = True,
    noise_level: float = 1.0,
) -> list[Experience]:
    """Generate training data using a Minimax agent.

    This is used to prime the network with reasonable moves before self-play starts.
    """
    agent = MinimaxAgent(depth=depth, randomize=randomize, noise_level=noise_level)
    all_experiences: list[Experience] = []

    print(f"Generating heuristic data for {num_games} games...")
    for i in range(num_games):
        if (i + 1) % 10 == 0:
            print(f"  Game {i+1}/{num_games}...")
        state = initial_state(config.board_size, config.walls_per_player)
        history = []

        while not state.is_terminal:
            # 1. Select best move using Minimax
            try:
                move = agent.select_move(state)
            except RuntimeError:
                break # No legal moves or error

            # 2. Record experience (policy target is 1.0 for the chosen move)
            # AlphaZero usually uses a full distribution, but for priming,
            # a one-hot distribution on a "good" move is a strong signal.
            policy_target = torch.zeros(3 * config.board_size * config.board_size, dtype=torch.float32)
            policy_target[move_to_index(move, config.board_size)] = 1.0

            history.append((encode_state(state), policy_target, state.current_player))

            # 3. Apply move
            state = apply_move(state, move)

            if len(history) > config.board_size * config.board_size * 2:
                # Prevent infinite games (though minimax shouldn't do this)
                break

        # 4. Finalise experiences with value targets
        if state.is_terminal:
            winner = state.winner
            for s_t, p_t, player in history:
                v = 1.0 if player == winner else -1.0
                all_experiences.append((s_t, p_t, v))
        else:
            # Draw or incomplete game
            for s_t, p_t, _ in history:
                all_experiences.append((s_t, p_t, 0.0))

    return all_experiences
