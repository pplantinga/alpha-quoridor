"""Self-play loop for data generation.

`run_self_play_games_batched` is the high-throughput entry point.
It drives *multiple* simultaneous MCTS trees and batches all network
inference into a single GPU forward pass per round of expansion.

`run_self_play_game` remains for API compatibility / single-game callers.
"""

import multiprocessing as mp
import random
from collections.abc import Generator

import torch

from game.board import initial_state
from game.rules import apply_move
from mcts.search import _SendType, _YieldType, run_mcts_generator
from model.network import QuoridorNet, encode_state, move_to_index
from training.buffer import Experience
from utils.config import Config

# ---------------------------------------------------------------------------
# Batched entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_self_play_games_batched(
    network: QuoridorNet,
    config: Config,
    num_games: int,
    device: torch.device = torch.device("cpu"),
) -> list[Experience]:
    """Generate `num_games` self-play games using batched GPU inference.

    All MCTS trees are advanced in lock-step: whenever any tree needs a
    network evaluation, we pause, gather ALL pending evaluations (up to
    `num_games` at once), run a single batched forward pass, and feed
    results back.

    Returns a flat list of Experience tuples from all games.
    """
    network.eval()
    n = config.board_size
    # Tighter move limit to prevent stalled games from wasting capacity
    max_moves = n * n * 2

    # Per-game state
    game_states = [initial_state(n, config.walls_per_player) for _ in range(num_games)]
    # history[i] accumulates (state_tensor, policy_target, current_player) for game i
    histories: list[list[tuple[torch.Tensor, torch.Tensor, int]]] = [[] for _ in range(num_games)]
    # state_counts[i] tracks how many times each QuoridorState was seen in game i
    from collections import Counter
    state_counts: list[Counter] = [Counter() for _ in range(num_games)]

    moves_played = [0] * num_games

    all_experiences: list[Experience] = []

    # -----------------------------------------------------------------------
    # Each game turn: run a complete MCTS search (batched), pick a move, advance
    # -----------------------------------------------------------------------
    active = list(range(num_games))  # indices of games still in progress

    while active:
        # --- Build one MCTS generator per active game, for this turn ---
        generators: dict[int, Generator[_YieldType, _SendType, dict] | None] = {}
        pending_encoded: dict[int, _YieldType] = {}   # game_idx -> tensor to evaluate

        for idx in active:
            state_counts[idx][game_states[idx]] += 1
            gen = run_mcts_generator(game_states[idx], config.mcts, training=True)
            generators[idx] = gen
            # Prime the generator — get its first yield
            try:
                pending_encoded[idx] = next(gen)
            except StopIteration:
                # Generator returned immediately (shouldn't happen on a fresh tree)
                generators[idx] = None  # type: ignore[assignment]

        probs_by_game: dict[int, dict] = {}

        # --- Drive all generators until they all finish ---
        while pending_encoded:
            # Batch all pending evaluations into one forward pass
            game_indices = list(pending_encoded.keys())
            batch = torch.stack(
                [pending_encoded[i] for i in game_indices], dim=0
            ).to(device)  # (K, 9, N, N)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                policy_batch, value_batch = network(batch)
            # policy_batch: (K, 3*N*N), value_batch: (K, 1)

            # Distribute results back and advance generators
            next_pending: dict[int, _YieldType] = {}
            for k, gi in enumerate(game_indices):
                result_to_send: _SendType = (
                    policy_batch[k].cpu(),
                    float(value_batch[k].item()),
                )
                try:
                    gen = generators[gi]
                    if gen is not None:
                        next_encoded = gen.send(result_to_send)
                        next_pending[gi] = next_encoded
                    else:
                        # Should not happen given current logic, but keeps ty happy
                        pass
                except StopIteration as e:
                    probs_by_game[gi] = e.value

            pending_encoded = next_pending

        # --- For each active game: record experience, pick move, advance state ---
        still_active = []
        for idx in active:
            probs = probs_by_game[idx]
            state = game_states[idx]

            # Record experience for this turn
            policy_target = torch.zeros(3 * n * n, dtype=torch.float32)
            for move, prob in probs.items():
                policy_target[move_to_index(move, n)] = prob
            histories[idx].append((encode_state(state), policy_target, state.current_player))

            # Sample action
            moves = list(probs.keys())
            action = random.choices(moves, weights=list(probs.values()), k=1)[0]
            new_state = apply_move(state, action)
            game_states[idx] = new_state
            moves_played[idx] += 1

            # Check for termination: winner, max moves, or 3-fold repetition
            is_repetition = state_counts[idx][new_state] >= 2 # 2 + the one we just added = 3

            if new_state.is_terminal or moves_played[idx] >= max_moves or is_repetition:
                # Finalise experiences for this game
                if not new_state.is_terminal:
                    # Timeout or repetition → draw
                    for s, p, _ in histories[idx]:
                        all_experiences.append((s, p, 0.0))
                else:
                    winner = new_state.winner
                    assert winner is not None


                    # Pre-calculate distances for the final state to handle the last move
                    final_encoded = encode_state(new_state)

                    for i, (s_t, p_t, player) in enumerate(histories[idx]):
                        # 1. Final outcome
                        outcome_v = 1.0 if player == winner else -1.0

                        # 2. Move penalty (encourages speed)
                        move_penalty = -0.002 * (len(histories[idx]) - i)

                        # 3. Progress reward (based on distance map change)
                        progress_reward = 0.0

                        # Get the "after" state for this move
                        if i < len(histories[idx]) - 1:
                            next_s_t = histories[idx][i+1][0]
                        else:
                            next_s_t = final_encoded

                        # In s_t (canonical), current player is at channel 0, distance map is channel 7
                        curr_pos_idx = torch.argmax(s_t[0].view(-1))
                        curr_dist = s_t[7].view(-1)[curr_pos_idx].item()

                        # In next_s_t (canonical), the player who moved is now the opponent!
                        # Their position is in channel 1, and their distance map is in channel 8.
                        next_pos_idx = torch.argmax(next_s_t[1].view(-1))
                        next_dist = next_s_t[8].view(-1)[next_pos_idx].item()

                        # dist_diff is positive if distance to goal decreased
                        dist_diff = curr_dist - next_dist
                        progress_reward = 0.01 * dist_diff

                        v = outcome_v + move_penalty + progress_reward
                        all_experiences.append((s_t, p_t, v))
            else:
                still_active.append(idx)

        active = still_active

    return all_experiences


# ---------------------------------------------------------------------------
# Legacy single-game wrapper (eval / play scripts / MCTSAgent)
# ---------------------------------------------------------------------------

def run_self_play_game(
    network: QuoridorNet,
    config: Config,
    device: torch.device = torch.device("cpu"),
) -> list[Experience]:
    """Convenience wrapper: run exactly one self-play game via the batched path."""
    return run_self_play_games_batched(network, config, num_games=1, device=device)


def _self_play_worker(
    network_state_dict: dict,
    config: Config,
    num_games: int,
    device_str: str,
    result_queue: mp.Queue,
):
    """Worker process for self-play."""
    device = torch.device(device_str)
    # Re-initialize network in the worker process
    network = QuoridorNet(config.board_size, config.model).to(device)
    network.load_state_dict(network_state_dict)

    experiences = run_self_play_games_batched(network, config, num_games, device)

    # Convert experiences to numpy for stable pickling/transfer
    picklable_exps = []
    for s, p, v in experiences:
        picklable_exps.append((s.numpy(), p.numpy(), v))

    result_queue.put(picklable_exps)
