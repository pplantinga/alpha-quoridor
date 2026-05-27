"""Quoridor game state representation."""

from __future__ import annotations

from dataclasses import dataclass

# (orientation, row, col) where orientation is 'h' (horizontal) or 'v' (vertical).
# A horizontal wall at (r, c) blocks movement between rows r and r+1
# for both col c and col c+1.
# A vertical wall at (r, c) blocks movement between cols c and c+1
# for both row r and row r+1.
Wall = tuple[str, int, int]

# ('move', row, col) or ('wall', row, col, orientation)
Move = tuple


@dataclass(frozen=True)
class QuoridorState:
    board_size: int
    walls_per_player: int
    # Positions as (row, col), 0-indexed. Row 0 is the top.
    player_pos: tuple[tuple[int, int], tuple[int, int]]
    walls_remaining: tuple[int, int]
    placed_walls: frozenset[Wall]
    current_player: int  # 0 or 1
    is_terminal: bool
    winner: int | None


def initial_state(board_size: int, walls_per_player: int) -> QuoridorState:
    """Return the starting state for a game with the given parameters.

    Player 0 starts at the bottom-centre (row board_size-1) and must reach row 0 (top).
    Player 1 starts at the top-centre (row 0) and must reach row board_size-1 (bottom).
    """
    mid = board_size // 2
    return QuoridorState(
        board_size=board_size,
        walls_per_player=walls_per_player,
        player_pos=((board_size - 1, mid), (0, mid)),
        walls_remaining=(walls_per_player, walls_per_player),
        placed_walls=frozenset(),
        current_player=0,
        is_terminal=False,
        winner=None,
    )


def goal_row(player: int, board_size: int) -> int:
    """Return the target row for the given player."""
    return 0 if player == 0 else board_size - 1
