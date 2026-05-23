"""Minimax Agent for Quoridor."""

import math

from game.board import Move, QuoridorState
from game.rules import a_star_path, apply_move, legal_moves


class MinimaxAgent:
    """Agent that selects moves using Minimax with Alpha-Beta pruning."""

    def __init__(self, depth: int = 2):
        self.depth = depth

    def select_move(self, state: QuoridorState) -> Move:
        """Select the best move using minimax search."""
        player = state.current_player
        moves = self._get_ordered_moves(state)

        if not moves:
            raise RuntimeError("No legal moves available.")

        best_move = moves[0]
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        for move in moves:
            next_state = apply_move(state, move)
            value = self._minimax(next_state, self.depth - 1, alpha, beta, False, player)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)

        return best_move

    def _minimax(
        self,
        state: QuoridorState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        original_player: int
    ) -> float:
        if depth == 0 or state.is_terminal:
            return self.evaluate(state, original_player)

        if maximizing_player:
            max_eval = -math.inf
            for move in self._get_ordered_moves(state):
                next_state = apply_move(state, move)
                eval = self._minimax(next_state, depth - 1, alpha, beta, False, original_player)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in self._get_ordered_moves(state):
                next_state = apply_move(state, move)
                eval = self._minimax(next_state, depth - 1, alpha, beta, True, original_player)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def _get_ordered_moves(self, state: QuoridorState) -> list[Move]:
        """Order moves to improve alpha-beta pruning efficiency."""
        moves = legal_moves(state)
        player = state.current_player

        path = a_star_path(state, player)
        if not path or len(path) < 2:
            return moves

        next_step = path[1]
        step_move: Move = ("move", next_step[0], next_step[1])

        ordered_moves = []
        if step_move in moves:
            ordered_moves.append(step_move)

        # Add other pawn moves
        for move in moves:
            if move[0] == "move" and move != step_move:
                ordered_moves.append(move)

        # Add wall placements
        for move in moves:
            if move[0] == "wall":
                ordered_moves.append(move)

        return ordered_moves

    def evaluate(self, state: QuoridorState, original_player: int) -> float:
        """Heuristic evaluation of the game state."""
        if state.is_terminal:
            if state.winner == original_player:
                return 10000.0
            else:
                return -10000.0

        p0_path = a_star_path(state, 0)
        p1_path = a_star_path(state, 1)

        # fallback to board size if no path found (should not happen in legal states)
        p0_len = len(p0_path) if p0_path else state.board_size * state.board_size
        p1_len = len(p1_path) if p1_path else state.board_size * state.board_size

        if original_player == 0:
            score = (p1_len - p0_len) * 10
        else:
            score = (p0_len - p1_len) * 10

        # Wall penalty: severe penalty for running out of walls
        if state.walls_remaining[original_player] == 0:
            score -= 20

        # Wall bonus: slight bonus for having more walls than opponent?
        # Actually user just said "penalty for zero walls left".

        return float(score)
