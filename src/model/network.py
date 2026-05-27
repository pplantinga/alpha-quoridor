"""Neural network architecture for Alpha Quoridor."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from game.board import Move, QuoridorState
from utils.config import ModelConfig


class ResBlock(nn.Module):
    """Residual block for the shared trunk."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class QuoridorNet(nn.Module):
    """Combined Policy and Value network for Quoridor."""

    def __init__(self, board_size: int, config: ModelConfig):
        super().__init__()
        self.board_size = board_size
        self.hidden_dim = config.hidden_dim

        # Input is 9 channels:
        # 0: P0 position
        # 1: P1 position
        # 2: Horizontal walls
        # 3: Vertical walls
        # 4: Current player turn (all 1s if P0, 0s if P1)
        # 5: P0 walls remaining (normalized)
        # 6: P1 walls remaining (normalized)
        # 7: P0 distance map (normalized)
        # 8: P1 distance map (normalized)
        self.input_conv = nn.Sequential(
            nn.Conv2d(9, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            *[ResBlock(self.hidden_dim) for _ in range(config.num_blocks)]
        )

        # Policy head: 3 channels of size (N, N)
        # Channel 0: Pawn move destinations
        # Channel 1: Horizontal wall placements
        # Channel 2: Vertical wall placements
        self.policy_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

        # Value head: Scalar in [-1, 1]
        self.value_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(8 * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Tensor of shape (B, 7, N, N)

        Returns:
            policy_logits: Tensor of shape (B, 3 * N * N)
            value: Tensor of shape (B, 1) in range [-1, 1]
        """
        b = x.size(0)
        x = self.input_conv(x)
        x = self.trunk(x)

        # Policy
        p = self.policy_conv(x)  # (B, 3, N, N)
        policy_logits = p.view(b, -1)  # (B, 3 * N * N)

        # Value
        v = self.value_conv(x)  # (B, 8, N, N)
        v = v.view(b, -1)
        value = self.value_fc(v)  # (B, 1)

        return policy_logits, value


def encode_state(state: QuoridorState) -> torch.Tensor:
    """Encode a QuoridorState into a 9-channel tensor.
    The representation is canonicalized such that the current player is always
    moving 'Up' (towards row 0).

    Returns:
        Tensor of shape (9, N, N)
    """
    import numpy as np
    n = state.board_size
    tensor_np = np.zeros((9, n, n), dtype=np.float32)

    cp = state.current_player
    opp = 1 - cp

    def flip_row(r):
        return (n - 1) - r if cp == 1 else r

    def flip_wall_row(r):
        return (n - 2) - r if cp == 1 else r

    # 0: Current player position, 1: Opponent position
    r_cp, c_cp = state.player_pos[cp]
    tensor_np[0, flip_row(r_cp), c_cp] = 1.0

    r_opp, c_opp = state.player_pos[opp]
    tensor_np[1, flip_row(r_opp), c_opp] = 1.0

    # 2: H walls, 3: V walls
    for orient, r, c in state.placed_walls:
        if orient == "h":
            tensor_np[2, flip_wall_row(r), c] = 1.0
        else:
            # Vertical wall spans rows r and r+1. Coordinates after flip: (n-1-r) and (n-2-r).
            # The top row of the flipped wall is n-2-r.
            v_row = (n - 2) - r if cp == 1 else r
            tensor_np[3, v_row, c] = 1.0

    # 4: Turn indicator (canonical: always set to 1.0 for the current player's perspective)
    tensor_np[4, :, :] = 1.0

    # 5, 6: Walls remaining normalized
    max_walls = max(1, state.walls_per_player)
    tensor_np[5, :, :] = state.walls_remaining[cp] / max_walls
    tensor_np[6, :, :] = state.walls_remaining[opp] / max_walls

    # 7, 8: Distance maps (high-performance bitboard version)
    # Channel 7: Current player distance map, Channel 8: Opponent distance map
    dist_maps = compute_distance_maps(state)
    tensor_np[7, :, :] = dist_maps[cp].numpy()
    tensor_np[8, :, :] = dist_maps[opp].numpy()

    # If we are Player 1, the distance maps also need to be flipped vertically
    if cp == 1:
        tensor_np[7, :, :] = np.flip(tensor_np[7, :, :], axis=0)
        tensor_np[8, :, :] = np.flip(tensor_np[8, :, :], axis=0)

    return torch.from_numpy(tensor_np)

def compute_distance_maps(state: QuoridorState) -> torch.Tensor:
    """Compute the shortest path distance from every cell to the goal rows.
    Returns:
        Tensor of shape (2, N, N) where channel 0 is P0's normalized distance map,
        and channel 1 is P1's normalized distance map.
    """
    import numpy as np

    from game.board import goal_row
    from game.rules import get_move_masks

    n = state.board_size
    dist_maps = np.full((2, n, n), float(n * n), dtype=np.float32)

    if n != 9:
        # Simple BFS fallback for non-standard board sizes
        return _compute_distance_maps_bfs(state)

    mask_n, mask_s, mask_e, mask_w = get_move_masks(state)

    for player in (0, 1):
        target_row = goal_row(player, n)
        goal_mask = ((1 << 9) - 1) << (target_row * 9)
        frontier = goal_mask
        visited = frontier
        d = 0.0

        # Mark distance 0 for the whole goal row
        dist_maps[player, target_row, :] = 0.0

        while frontier:
            d += 1.0
            # Expand using bitboard masks
            # Note: We want to expand FROM the goal row to all cells.
            # Moving from cell i to i-9 (North) is possible if mask_n[i] is set.
            # If we are expanding FROM goal row (South-to-North), we use mask_s on the results?
            # Wait, if we can move North from i to i-9, then i-9 is reachable from i.
            # If frontier is at row r, frontier << 9 is at row r+1.
            # This is possible if bits in r+1 allow North move (mask_n).
            # No, if bits in r allow South move (mask_s).
            # Let's keep it simple:
            next_frontier = ((frontier & mask_n) >> 9)
            next_frontier |= ((frontier & mask_s) << 9)
            next_frontier |= ((frontier & mask_w) >> 1)
            next_frontier |= ((frontier & mask_e) << 1)

            frontier = next_frontier & ~visited
            if not frontier:
                break

            # Update distance map for newly reached cells
            indices = [i for i in range(81) if (frontier >> i) & 1]
            for i in indices:
                dist_maps[player, i // 9, i % 9] = d

            visited |= frontier
    return torch.from_numpy(dist_maps / float(n))


def _compute_distance_maps_bfs(state: QuoridorState) -> torch.Tensor:
    """Fallback BFS implementation for non-9x9 boards."""
    from collections import deque

    import numpy as np

    from game.board import goal_row
    from game.rules import _DIRECTIONS, is_blocked

    n = state.board_size
    dist_maps = np.full((2, n, n), float(n * n), dtype=np.float32)

    for player in (0, 1):
        target_row = goal_row(player, n)
        queue = deque()
        for c in range(n):
            queue.append((target_row, c, 0))
            dist_maps[player, target_row, c] = 0.0

        while queue:
            r, c, d = queue.popleft()
            for direction_name, (dr, dc) in _DIRECTIONS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    if not is_blocked(state, (r, c), direction_name):
                            if dist_maps[player, nr, nc] > d + 1:
                                dist_maps[player, nr, nc] = d + 1.0
                                queue.append((nr, nc, d + 1))

    return torch.from_numpy(dist_maps / float(n))


def move_to_index(move: Move, board_size: int) -> int:
    """Convert a Move tuple to a flat action index in [0, 3*N*N - 1]."""
    n = board_size
    if move[0] == "move":
        _, r, c = move
        return 0 * n * n + r * n + c
    elif move[0] == "wall":
        _, r, c, orient = move
        channel = 1 if orient == "h" else 2
        return channel * n * n + r * n + c
    raise ValueError(f"Unknown move type: {move}")


def index_to_move(index: int, board_size: int) -> Move:
    """Convert a flat action index back to a Move tuple."""
    n = board_size
    channel = index // (n * n)
    rem = index % (n * n)
    r = rem // n
    c = rem % n

    if channel == 0:
        return ("move", r, c)
    elif channel == 1:
        return ("wall", r, c, "h")
    elif channel == 2:
        return ("wall", r, c, "v")
    raise ValueError(f"Index {index} out of bounds for board size {n}")
