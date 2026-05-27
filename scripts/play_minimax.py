"""Interactive CLI game against Minimax agent."""

import argparse
import sys
from pathlib import Path

# Add current dir to path for play_human import
sys.path.append(str(Path(__file__).parent))
from play_human import (
    parse_input,
    print_status,
    render_board,
)

from agent.minimax_agent import MinimaxAgent
from game.board import initial_state
from game.rules import apply_move, legal_moves


def get_human_move(state) -> tuple:
    valid_moves = legal_moves(state)
    while True:
        user_input = input(f"P{state.current_player} (Human)> ")
        if user_input.strip().lower() in ("q", "quit"):
            print("Game aborted.")
            sys.exit(0)

        if user_input.strip().lower() == "moves":
            print("Legal moves:")
            for m in valid_moves:
                print(f"  {m}")
            continue

        parsed = parse_input(user_input)
        if not parsed:
            print("Invalid input format!")
            continue

        move_to_apply = None
        if parsed[0] == "dir":
            dr, dc = {
                "N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1),
                "NE": (-1, 1), "NW": (-1, -1), "SE": (1, 1), "SW": (1, -1),
            }[parsed[1]]
            r, c = state.player_pos[state.current_player]
            target_r, target_c = r + dr, c + dc

            for m in valid_moves:
                if m[0] == "move":
                    if m[1] == target_r and m[2] == target_c:
                        move_to_apply = m
                        break

                    if parsed[1] in ("NE", "NW", "SE", "SW"):
                        if (m[1] - r) * dr > 0 and (m[2] - c) * dc > 0:
                            move_to_apply = m
                            break
                    else:
                        if target_r != r and target_r - r > 0 and m[1] > r and m[2] == c and parsed[1] == "S":
                            move_to_apply = m
                            break
                        if target_r != r and target_r - r < 0 and m[1] < r and m[2] == c and parsed[1] == "N":
                            move_to_apply = m
                            break
                        if target_c != c and target_c - c > 0 and m[2] > c and m[1] == r and parsed[1] == "E":
                            move_to_apply = m
                            break
                        if target_c != c and target_c - c < 0 and m[2] < c and m[1] == r and parsed[1] == "W":
                            move_to_apply = m
                            break

            if not move_to_apply:
                print(f"Cannot move {parsed[1]}! (blocked or illegal)")
                continue

        elif parsed[0] == "wall":
            move_to_apply = parsed
            if move_to_apply not in valid_moves:
                print("Illegal wall placement! (blocked, overlaps, or cuts off path)")
                continue

        if move_to_apply:
            return move_to_apply


def main() -> None:
    parser = argparse.ArgumentParser("Human vs Minimax Quoridor")
    parser.add_argument("--size", type=int, default=9, help="Board size")
    parser.add_argument("--walls", type=int, default=10, help="Walls per player")
    parser.add_argument("--depth", type=int, default=2, help="Minimax search depth")
    parser.add_argument("--ai_player", type=int, default=1, choices=[0, 1], help="Which player AI plays (0 or 1)")
    args = parser.parse_args()

    minimax_agent = MinimaxAgent(depth=args.depth)
    state = initial_state(args.size, args.walls)

    while not state.is_terminal:
        render_board(state)

        if state.current_player == args.ai_player:
            print(f"Minimax AI (Player {args.ai_player}, depth={args.depth}) is thinking...")
            move = minimax_agent.select_move(state)
            print(f"AI plays: {move}")
        else:
            print_status(state)
            move = get_human_move(state)

        state = apply_move(state, move)

    # Game over
    render_board(state)
    print(f"Game Over! Player {state.winner} wins!")


if __name__ == "__main__":
    main()
