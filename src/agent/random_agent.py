"""Random Agent."""

import random

from game.board import Move, QuoridorState
from game.rules import legal_moves


class RandomAgent:
    """Agent that selects a move uniformly at random from legal moves."""

    def select_move(self, state: QuoridorState) -> Move:
        moves = legal_moves(state)
        if not moves:
            raise RuntimeError("No legal moves available.")
        return random.choice(moves)
