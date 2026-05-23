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

    Returns:
        Tensor of shape (9, N, N)
    """
    import numpy as np
    n = state.board_size
    # Use numpy for faster initialization and manipulation
    tensor_np = np.zeros((9, n, n), dtype=np.float32)

    # 0: P0 position
    r0, c0 = state.player_pos[0]
    tensor_np[0, r0, c0] = 1.0

    # 1: P1 position
    r1, c1 = state.player_pos[1]
    tensor_np[1, r1, c1] = 1.0

    # 2: H walls
    # 3: V walls
    for orient, r, c in state.placed_walls:
        if orient == "h":
            tensor_np[2, r, c] = 1.0
        else:
            tensor_np[3, r, c] = 1.0

    # 4: Current player turn (all 1s for P0, 0s for P1)
    if state.current_player == 0:
        tensor_np[4, :, :] = 1.0

    # 5: P0 walls normalized
    max_walls = max(1, state.walls_per_player)
    tensor_np[5, :, :] = state.walls_remaining[0] / max_walls

    # 6: P1 walls normalized
    tensor_np[6, :, :] = state.walls_remaining[1] / max_walls

    # 7, 8: Distance maps
    tensor_np[7:9, :, :] = compute_distance_maps(state).numpy()

    return torch.from_numpy(tensor_np)

def compute_distance_maps(state: QuoridorState) -> torch.Tensor:
    """Compute the shortest path distance from every cell to the goal rows.
    Returns:
        Tensor of shape (2, N, N) where channel 0 is P0's normalized distance map,
        and channel 1 is P1's normalized distance map.
    """
    from collections import deque

    import numpy as np

    from game.board import goal_row
    from game.rules import _DIRECTIONS, is_blocked

    n = state.board_size
    # Use numpy for the distance map calculations
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
                    # A wall blocking going from (r,c) to (nr,nc) means it blocks (nr,nc) to (r,c)
                    if not is_blocked(state, (r, c), direction_name):
                        if dist_maps[player, nr, nc] > d + 1:
                            dist_maps[player, nr, nc] = d + 1.0
                            queue.append((nr, nc, d + 1))

    return torch.from_numpy(dist_maps / float(n * n))


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
