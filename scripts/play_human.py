"""Interactive human-vs-human CLI for Alpha Quoridor."""

import argparse
import sys

from game.board import initial_state
from game.rules import apply_move, legal_moves


def render_board(state) -> None:
    """Render the Quoridor board in ASCII."""
    n = state.board_size

    # Intersection col headers (spaced to align over the + symbols)
    col_labels = "       " + "   ".join(chr(ord('a') + c) for c in range(n - 1))
    print("\n" + col_labels)

    for row in range(n):
        # Print cell row
        line = "    "
        for col in range(n):
            if (row, col) == state.player_pos[0]:
                line += " 0 "
            elif (row, col) == state.player_pos[1]:
                line += " 1 "
            else:
                line += " . "

            # vertical border
            if col < n - 1:
                if ("v", row, col) in state.placed_walls or ("v", row - 1, col) in state.placed_walls:
                    line += "║"
                else:
                    line += "|"
        print(line)

        # Print intersection row (walls)
        if row < n - 1:
            line = f" {row:<2} "
            for col in range(n):
                if ("h", row, col) in state.placed_walls or ("h", row, col - 1) in state.placed_walls:
                    line += "==="
                else:
                    line += "---"

                if col < n - 1:
                    if ("h", row, col) in state.placed_walls:
                        line += "="
                    elif ("v", row, col) in state.placed_walls:
                        line += "║"
                    else:
                        line += "+"
            print(line)
    print()


def print_status(state) -> None:
    p = state.current_player
    print(f"Player {p}'s turn. Walls remaining: P0: {state.walls_remaining[0]} | P1: {state.walls_remaining[1]}")
    print("Move formats:")
    print("  Direction: 'N', 'S', 'E', 'W' (or 'NE', 'NW', 'SE', 'SW' for lateral jumps)")
    print("  Wall: 'a0h' or 'b2v' (col char, row num, orient 'h'/'v')")
    print("  Other: 'q' to quit, 'moves' to list legal moves")


def parse_input(user_input: str) -> tuple | None:
    """Parse user input into a move tuple, or return None if invalid format."""
    cmd = user_input.strip().lower()
    if not cmd:
        return None

    if cmd in ("n", "s", "e", "w", "ne", "nw", "se", "sw"):
        # We don't know the exact destination yet just from direction,
        # but to map it to our ('move', r, c) representation we need the current pos.
        # So we just return a temporary dictation and let the game loop resolve it.
        return ("dir", cmd.upper())

    if len(cmd) >= 3 and cmd[0].isalpha() and cmd[-1] in ('h', 'v'):
        row_str = cmd[1:-1]
        col_char = cmd[0]
        orient = cmd[-1]
        if row_str.isdigit():
            c = ord(col_char) - ord('a')
            r = int(row_str)
            return ("wall", r, c, orient)

    return None


def main() -> None:
    parser = argparse.ArgumentParser("Human vs Human Quoridor")
    parser.add_argument("--size", type=int, default=5, help="Board size")
    parser.add_argument("--walls", type=int, default=3, help="Walls per player")
    args = parser.parse_args()

    state = initial_state(args.size, args.walls)

    while not state.is_terminal:
        render_board(state)
        print_status(state)

        valid_moves = legal_moves(state)

        user_input = input(f"P{state.current_player}> ")
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

        # Resolve direction to actual move
        move_to_apply = None
        if parsed[0] == "dir":
            # Find the legal move that goes in that direction
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
                    # Handle jumps (straight or diagonal)
                    # Ensure we are checking the direction vector
                    if parsed[1] in ("NE", "NW", "SE", "SW"):
                        # Diagonal jump: row and col must move in the correct direction
                        if (m[1] - r) * dr > 0 and (m[2] - c) * dc > 0:
                            move_to_apply = m
                            break
                    else:
                        # Straight jump
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

        # Apply move
        if move_to_apply:
            state = apply_move(state, move_to_apply)

    # Game over
    render_board(state)
    print(f"Game Over! Player {state.winner} wins!")


if __name__ == "__main__":
    main()
