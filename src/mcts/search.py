"""Monte Carlo Tree Search implementation.

The core search logic is exposed as a generator (`run_mcts_generator`) so that an
external batch-manager can collect pending network evaluations from many concurrent
MCTS trees and push them through the GPU in a single batched forward pass.

Legacy wrapper `run_mcts` is kept for single-game callers (eval, play scripts).
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from game.board import Move, QuoridorState
from game.rules import legal_moves
from mcts.node import MCTSNode
from model.network import QuoridorNet, encode_state, move_to_index
from utils.config import MCTSConfig

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Generator-based MCTS — one generator per game/tree
# ---------------------------------------------------------------------------

# Type alias for the value yielded to the batch manager
_YieldType = torch.Tensor  # encoded state (9, N, N)
# Type sent back by the batch manager: (flat policy_logits tensor, scalar value)
_SendType = tuple[torch.Tensor, float]


def run_mcts_generator(
    initial_state: QuoridorState,
    config: MCTSConfig,
    training: bool = False,
) -> Generator[_YieldType, _SendType, dict[Move, float]]:
    """Generator-based MCTS.

    Yields an encoded state tensor whenever it needs a network evaluation.
    The driver should `.send((policy_logits, value))` back.

    Returns the visit-probability dict when all simulations are complete.
    """
    root = MCTSNode(state=initial_state)

    # ------------------------------------------------------------------
    # Expand root (first network call)
    # ------------------------------------------------------------------
    value = yield from _expand_generator(root)

    if training and root.is_expanded:
        _add_dirichlet_noise(root, config.dirichlet_noise_alpha, config.dirichlet_noise_epsilon)

    # ------------------------------------------------------------------
    # MCTS simulations
    # ------------------------------------------------------------------
    for _ in range(config.num_simulations):
        node = root
        search_path: list[MCTSNode] = [node]

        # 1. Selection
        # Continue as long as the node has been expanded (we have priors for its children)
        while node.is_expanded and not node.state.is_terminal:
            _, node = node.select_child(config.c_puct)
            search_path.append(node)

        # 2. Expansion / Evaluation
        if node.state.is_terminal:
            assert node.state.winner is not None
            value = 1.0 if node.state.winner == node.state.current_player else -1.0
        else:
            value = yield from _expand_generator(node)

        # 3. Backup
        _backup(search_path, value)

    # ------------------------------------------------------------------
    # Build visit-probability distribution
    # ------------------------------------------------------------------
    visits = {action: child.visit_count for action, child in root.children.items()}

    total_visits = sum(visits.values())
    if total_visits == 0:
        moves = legal_moves(root.state)
        prob = 1.0 / len(moves) if moves else 1.0
        return {m: prob for m in moves}

    if training:
        temp = config.temperature
        if temp == 0.0:
            best = max(visits, key=visits.__getitem__)
            return {m: 1.0 if m == best else 0.0 for m in visits}
        else:
            v_adj = {m: v ** (1.0 / temp) for m, v in visits.items()}
            total = sum(v_adj.values())
            return {m: v / total for m, v in v_adj.items()}
    else:
        best = max(visits, key=visits.__getitem__)
        return {m: 1.0 if m == best else 0.0 for m in visits}


def _expand_generator(
    node: MCTSNode,
) -> Generator[_YieldType, _SendType, float]:
    """Sub-generator that yields an encoded-state tensor and expands the node.

    Returns the network value for the node.
    """
    valid_moves = legal_moves(node.state)
    if not valid_moves:
        return 0.0

    # Yield the raw encoded tensor (no batch dim, no device move — caller handles that)
    encoded = encode_state(node.state)
    policy_logits, value = yield encoded  # <-- suspension point

    board_size = node.state.board_size
    valid_indices = [move_to_index(m, board_size) for m in valid_moves]
    logits = policy_logits[valid_indices]
    probs = F.softmax(logits, dim=0).numpy()

    for idx, move in enumerate(valid_moves):
        node.child_priors[move] = float(probs[idx])

    node.is_expanded = True
    return float(value)


# ---------------------------------------------------------------------------
# Legacy single-game wrapper (used by eval/play scripts and MCTSAgent)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_mcts(
    initial_state: QuoridorState,
    network: QuoridorNet,
    config: MCTSConfig,
    device: torch.device = torch.device("cpu"),
    training: bool = False,
) -> dict[Move, float]:
    """Run MCTS for a single game, evaluating nodes one-at-a-time (legacy path).

    For high-throughput training, prefer the batched path in self_play.py.
    """
    network.eval()
    gen = run_mcts_generator(initial_state, config, training=training)

    result: dict[Move, float] | None = None
    value_to_send: _SendType | None = None

    try:
        while True:
            if value_to_send is None:
                encoded = next(gen)
            else:
                encoded = gen.send(value_to_send)

            # Single-item forward pass
            with torch.no_grad():
                t = encoded.unsqueeze(0).to(device)
                policy_logits, val_tensor = network(t)
            value_to_send = (policy_logits[0].cpu(), float(val_tensor.item()))

    except StopIteration as e:
        result = e.value

    assert result is not None
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backup(search_path: list[MCTSNode], value: float) -> None:
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        value = -value


def _add_dirichlet_noise(node: MCTSNode, alpha: float, epsilon: float) -> None:
    actions = list(node.child_priors.keys())
    if not actions:
        return
    noise = np.random.dirichlet([alpha] * len(actions))
    for idx, action in enumerate(actions):
        node.child_priors[action] = (1 - epsilon) * node.child_priors[action] + epsilon * noise[idx]
        # Also update the actual child if it happens to already exist (rare at root expansion)
        if action in node.children:
            node.children[action].prior = node.child_priors[action]
