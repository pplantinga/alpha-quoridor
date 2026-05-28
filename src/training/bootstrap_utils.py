"""Utilities for bootstrapping Alpha Quoridor training."""

import multiprocessing as mp

import torch

from agent.minimax_agent import MinimaxAgent
from game.board import initial_state
from game.rules import apply_move
from model.network import encode_state, move_to_index
from training.buffer import Experience
from training.reward_shaping import step_shaping
from utils.config import Config


def _generate_single_heuristic_game(args) -> list[Experience]:
    """Helper for multiprocessing."""
    config, depth, randomize, noise_level = args
    rw = config.reward
    agent = MinimaxAgent(depth=depth, randomize=randomize, noise_level=noise_level)
    state = initial_state(config.board_size, config.walls_per_player)
    history: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    while not state.is_terminal:
        try:
            move = agent.select_move(state)
        except RuntimeError:
            break

        policy_target = torch.zeros(3 * config.board_size * config.board_size, dtype=torch.float32)
        policy_target[move_to_index(move, config.board_size)] = 1.0
        history.append((encode_state(state), policy_target, state.current_player))
        state = apply_move(state, move)

        if len(history) > config.board_size * config.board_size * 2:
            break

    picklable_exps: list[tuple] = []
    if state.is_terminal:
        winner = state.winner
        final_encoded = encode_state(state)
        for i, (s_t, p_t, player) in enumerate(history):
            outcome_v = 1.0 if player == winner else -1.0
            # Apply the same minimax-style shaping used in self-play
            next_enc = history[i + 1][0] if i + 1 < len(history) else final_encoded
            shaping = step_shaping(s_t, next_enc, rw.progress_weight, rw.block_weight)
            v = float(outcome_v + shaping)
            v = max(-1.0, min(1.0, v))
            picklable_exps.append((s_t.numpy(), p_t.numpy(), v))
    else:
        # Timed-out game: mirror the draw handling from self_play
        for s_t, p_t, _ in history:
            from training.reward_shaping import heuristic_value
            h = heuristic_value(s_t)
            v = float(rw.draw_penalty + rw.draw_heuristic_w * h)
            v = max(-1.0, min(1.0, v))
            picklable_exps.append((s_t.numpy(), p_t.numpy(), v))
    return picklable_exps


def generate_heuristic_data(
    config: Config,
    num_games: int,
    depth: int = 2,
    randomize: bool = True,
    noise_level: float = 1.0,
) -> list[Experience]:
    """Generate training data using a Minimax agent in parallel."""
    all_experiences: list[Experience] = []

    print(f"Generating heuristic data for {num_games} games in parallel...")

    # Use spawn to avoid issues with CUDA if called from a script that initialized it
    # although Minimax typically doesn't use CUDA.
    ctx = mp.get_context("spawn")

    worker_args = [(config, depth, randomize, noise_level) for _ in range(num_games)]

    with ctx.Pool() as pool:
        results = pool.map(_generate_single_heuristic_game, worker_args)

    for res in results:
        for s_np, p_np, v in res:
            all_experiences.append((
                torch.from_numpy(s_np),
                torch.from_numpy(p_np),
                v
            ))

    return all_experiences

