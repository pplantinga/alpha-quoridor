"""MCTS Node definition."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from game.board import Move, QuoridorState


@dataclass
class MCTSNode:
    """A node in the Monte Carlo Tree Search."""

    state: QuoridorState
    parent: MCTSNode | None = None
    move: Move | None = None  # The move that led to this node
    children: dict[Move, MCTSNode] = field(default_factory=dict)
    child_priors: dict[Move, float] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Return the expected value (average value) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_leaf(self) -> bool:
        """Return True if this node has no children."""
        return not bool(self.children)

    def select_child(self, c_puct: float) -> tuple[Move, MCTSNode]:
        """Select the child with the highest PUCT score."""
        best_score = -float("inf")
        best_move = None

        # We iterate over all moves that the network gave priors for
        for move, prior in self.child_priors.items():
            child = self.children.get(move)
            if child is not None:
                q = -child.q_value
                v = child.visit_count
            else:
                q = 0.0
                v = 0

            u = c_puct * prior * math.sqrt(self.visit_count) / (1 + v)
            score = q + u

            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:
            raise RuntimeError("Called select_child on a node with no child_priors.")

        # If the selected child doesn't exist yet, create it lazily
        if best_move not in self.children:
            from game.rules import apply_move
            next_state = apply_move(self.state, best_move)
            self.children[best_move] = MCTSNode(
                state=next_state,
                parent=self,
                move=best_move,
                prior=self.child_priors[best_move],
            )

        return best_move, self.children[best_move]
