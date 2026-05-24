"""Evaluation script — pit trained model against minimax agent."""

import argparse
from pathlib import Path

import torch

from agent.mcts_agent import MCTSAgent
from agent.minimax_agent import MinimaxAgent
from game.board import initial_state
from game.rules import apply_move
from model.network import QuoridorNet
from utils.checkpoint import load_model_weights
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Alpha Quoridor")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--depth", type=int, default=2, help="Minimax depth for baseline")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = QuoridorNet(config.board_size, config.model).to(device)
    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_model_weights(network, checkpoint["model_state_dict"])
        print(f"Loaded {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Using untrained network!")

    mcts_agent = MCTSAgent(network, config.mcts, device)
    baseline_agent = MinimaxAgent(depth=args.depth)

    mcts_wins = 0
    baseline_wins = 0
    draws = 0

    print(f"Playing {args.games} games against Minimax (depth={args.depth})...")
    for game in range(args.games):
        state = initial_state(config.board_size, config.walls_per_player)
        move_count = 0

        # Alternate who goes first
        agents = {
            0: mcts_agent if game % 2 == 0 else baseline_agent,
            1: baseline_agent if game % 2 == 0 else mcts_agent,
        }

        while not state.is_terminal and move_count < 200:
            agent = agents[state.current_player]

            # MCTS agent uses training=False for deterministic max-visit move selection
            if isinstance(agent, MCTSAgent):
                move = agent.select_move(state, training=False)
            else:
                move = agent.select_move(state)

            state = apply_move(state, move)
            move_count += 1

        winner = state.winner
        if winner is not None:
            print(f"Game {game + 1}/{args.games} — Winner: Player {winner} ({'MCTS' if agents[winner] is mcts_agent else 'Minimax'}) in {move_count} moves")
            if agents[winner] is mcts_agent:
                mcts_wins += 1
            else:
                baseline_wins += 1
        else:
            print(f"Game {game + 1}/{args.games} — Result: Draw in {move_count} moves")
            draws += 1

    print("\n--- Results ---")
    print(f"MCTS Agent: {mcts_wins}")
    print(f"Minimax Baseline: {baseline_wins}")
    print(f"Draws: {draws}")


if __name__ == "__main__":
    main()
