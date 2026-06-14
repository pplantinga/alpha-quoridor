"""Left-right symmetry augmentation for Quoridor experience tuples.

The Quoridor board is left-right symmetric: a game played on one side of the
vertical center axis is strategically equivalent to its mirror image.  Every
encoded (state, policy, value) triple therefore has a valid mirror twin that
can be added to the replay buffer for free, doubling effective dataset size.

Encoding conventions (from model/network.py):
    State  (9, N, N):
        ch 0  – current-player pawn position (one-hot, canonical coords)
        ch 1  – opponent pawn position (one-hot, canonical coords)
        ch 2  – horizontal wall map  – wall at (r, c) spans cols c and c+1
        ch 3  – vertical wall map    – wall at (r, c) spans rows r and r+1
        ch 4  – turn indicator (scalar broadcast, unchanged by horizontal flip)
        ch 5  – current-player walls remaining (scalar broadcast)
        ch 6  – opponent walls remaining (scalar broadcast)
        ch 7  – current-player distance map
        ch 8  – opponent distance map

    Policy (3*N*N flat):
        [0       .. N²-1]   pawn-move channel    – pawn dest at (r, c)
        [N²      .. 2N²-1]  h-wall channel       – h-wall at (r, c)
        [2N²     .. 3N²-1]  v-wall channel       – v-wall at (r, c)

Left-right flip rules:
    Pawn positions:   col c  →  n-1-c   (standard mirror)
    Wall positions:   col c  →  n-2-c   (wall spans c..c+1; mirror of c..c+1
                                          is n-2-c..n-1-c, so new anchor = n-2-c)

    For the state tensor:
        ch 0,1 (pawn one-hots)    – flip full N×N slice along col axis
        ch 2,3 (wall maps)        – flip only the valid n-1 columns, pad right with zero
        ch 4,5,6 (scalars)        – unchanged (horizontally uniform)
        ch 7,8 (distance maps)    – flip full N×N slice along col axis (symmetric by BFS)

    For the policy tensor (reshaped to (3, N, N)):
        ch 0 (pawn moves)         – flip full N×N along col axis
        ch 1,2 (wall placements)  – flip valid n-1 columns, pad right with zero
"""

from __future__ import annotations

import torch


def flip_state(state: torch.Tensor) -> torch.Tensor:
    """Return the left-right mirrored encoding of a canonical state tensor.

    Args:
        state: Float tensor of shape (9, N, N).

    Returns:
        Float tensor of shape (9, N, N) representing the mirrored position.
    """
    n = state.shape[1]
    flipped = state.clone()

    # Pawn channels – standard horizontal flip
    flipped[0] = state[0].flip(dims=[1])
    flipped[1] = state[1].flip(dims=[1])

    # Wall channels – valid cols are 0..n-2; mirror within that range
    flipped[2] = _flip_wall_channel(state[2], n)
    flipped[3] = _flip_wall_channel(state[3], n)

    # Scalar channels (4, 5, 6) – horizontally uniform, copy as-is (already in flipped)

    # Distance map channels – standard horizontal flip
    flipped[7] = state[7].flip(dims=[1])
    flipped[8] = state[8].flip(dims=[1])

    return flipped


def flip_policy(policy: torch.Tensor, board_size: int) -> torch.Tensor:
    """Return the left-right mirrored policy target vector.

    Args:
        policy: Float tensor of shape (3 * board_size * board_size,).
        board_size: Board side length N.

    Returns:
        Float tensor of the same shape with mirrored move probabilities.
    """
    n = board_size
    p = policy.view(3, n, n)
    flipped = torch.zeros_like(p)

    # Pawn move channel – standard horizontal flip
    flipped[0] = p[0].flip(dims=[1])

    # Wall channels – mirror within valid columns (0..n-2)
    flipped[1] = _flip_wall_channel(p[1], n)
    flipped[2] = _flip_wall_channel(p[2], n)

    return flipped.view(-1)


def flip_experience(
    state: torch.Tensor,
    policy: torch.Tensor,
    value: float,
    board_size: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return the mirrored (state, policy, value) triple.

    Value is unchanged because the game outcome is symmetric under left-right
    reflection.
    """
    return flip_state(state), flip_policy(policy, board_size), value


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _flip_wall_channel(channel: torch.Tensor, n: int) -> torch.Tensor:
    """Mirror a wall channel's valid columns (0..n-2) and return an N×N tensor.

    Walls at column c (spanning c and c+1) map to column n-2-c after the
    left-right flip.  The column at index n-1 is always zero (no wall can
    start at the last column), so we operate on the n-1 valid columns, flip
    them, and re-pad with a zero column on the right.
    """
    valid = channel[:, :n - 1]           # (N, N-1) — drop always-zero last col
    flipped_valid = valid.flip(dims=[1]) # reverse within valid range
    pad = torch.zeros(channel.shape[0], 1, dtype=channel.dtype, device=channel.device)
    return torch.cat([flipped_valid, pad], dim=1)  # (N, N)
