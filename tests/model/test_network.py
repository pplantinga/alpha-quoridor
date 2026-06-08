"""Tests for the model architecture and state encoding."""

import torch

from game.board import QuoridorState, initial_state
from model.network import (
    QuoridorNet,
    canonical_index_to_move,
    canonical_move_to_index,
    encode_state,
    index_to_move,
    move_to_index,
)
from utils.config import ModelConfig


def test_encode_state_shape() -> None:
    state = initial_state(5, 3)
    tensor = encode_state(state)
    assert tensor.shape == (9, 5, 5)


def test_encode_state_values() -> None:
    state = QuoridorState(
        board_size=5,
        walls_per_player=3,
        player_pos=((4, 2), (0, 2)),
        walls_remaining=(2, 3),
        placed_walls=frozenset({("h", 1, 1), ("v", 2, 2)}),
        current_player=0,
        is_terminal=False,
        winner=None,
    )
    tensor = encode_state(state)

    # Channel 0: Player 0 pos
    assert tensor[0, 4, 2] == 1.0
    assert tensor[0].sum() == 1.0

    # Channel 1: Player 1 pos
    assert tensor[1, 0, 2] == 1.0
    assert tensor[1].sum() == 1.0

    # Channel 2: H walls
    assert tensor[2, 1, 1] == 1.0
    assert tensor[2].sum() == 1.0

    # Channel 3: V walls
    assert tensor[3, 2, 2] == 1.0
    assert tensor[3].sum() == 1.0

    # Channel 4: Current player
    # Player 0 turn -> all 1s
    assert (tensor[4] == 1.0).all()

    # Channel 5: P0 walls normalized (2/3)
    assert torch.allclose(tensor[5], torch.tensor(2 / 3))

    # Channel 6: P1 walls normalized (3/3)
    assert torch.allclose(tensor[6], torch.tensor(1.0))


def test_network_forward() -> None:
    board_size = 5
    config = ModelConfig(hidden_dim=32, num_blocks=2)
    net = QuoridorNet(board_size, config)

    batch_size = 4
    x = torch.zeros((batch_size, 9, board_size, board_size))
    policy_logits, value = net(x)

    # 3 channels (move, h_wall, v_wall) * N * N
    assert policy_logits.shape == (batch_size, 3 * 25)

    # Value scalar
    assert value.shape == (batch_size, 1)

    # Tanh bounds
    assert (value >= -1).all()
    assert (value <= 1).all()


def test_move_conversion() -> None:
    n = 5

    # Pawn move
    m1 = ("move", 2, 3)
    idx1 = move_to_index(m1, n)
    assert idx1 == 0 * 25 + 2 * 5 + 3
    assert index_to_move(idx1, n) == m1

    # H wall
    m2 = ("wall", 1, 4, "h")
    idx2 = move_to_index(m2, n)
    assert idx2 == 1 * 25 + 1 * 5 + 4
    assert index_to_move(idx2, n) == m2

    # V wall
    m3 = ("wall", 0, 0, "v")
    idx3 = move_to_index(m3, n)
    assert idx3 == 2 * 25 + 0 * 5 + 0
    assert index_to_move(idx3, n) == m3


def test_canonical_move_conversion() -> None:
    n = 5

    # Player 0 should be identical to raw
    pawn_p0 = ("move", 2, 3)
    assert canonical_move_to_index(pawn_p0, n, 0) == move_to_index(pawn_p0, n)
    assert canonical_index_to_move(canonical_move_to_index(pawn_p0, n, 0), n, 0) == pawn_p0

    # Player 1 pawn move: row 1 on board size 5 becomes canonical row (5 - 1 - 1) = 3.
    pawn_p1 = ("move", 1, 3)
    idx_pawn_p1 = canonical_move_to_index(pawn_p1, n, 1)
    assert idx_pawn_p1 == 0 * 25 + 3 * 5 + 3
    assert canonical_index_to_move(idx_pawn_p1, n, 1) == pawn_p1

    # Player 1 horizontal wall: row 1 on board size 5 becomes canonical row (5 - 2 - 1) = 2.
    wall_h_p1 = ("wall", 1, 2, "h")
    idx_wall_h_p1 = canonical_move_to_index(wall_h_p1, n, 1)
    assert idx_wall_h_p1 == 1 * 25 + 2 * 5 + 2
    assert canonical_index_to_move(idx_wall_h_p1, n, 1) == wall_h_p1

    # Player 1 vertical wall: row 0 on board size 5 becomes canonical row (5 - 2 - 0) = 3.
    wall_v_p1 = ("wall", 0, 1, "v")
    idx_wall_v_p1 = canonical_move_to_index(wall_v_p1, n, 1)
    assert idx_wall_v_p1 == 2 * 25 + 3 * 5 + 1
    assert canonical_index_to_move(idx_wall_v_p1, n, 1) == wall_v_p1

    # Roundtrip check for all possible actions for both players
    for p in (0, 1):
        # Pawn moves
        for r in range(n):
            for c in range(n):
                m = ("move", r, c)
                idx = canonical_move_to_index(m, n, p)
                assert canonical_index_to_move(idx, n, p) == m

        # Walls (horizontal/vertical on a grid of size (n-1)x(n-1))
        for r in range(n - 1):
            for c in range(n - 1):
                for orient in ("h", "v"):
                    m = ("wall", r, c, orient)
                    idx = canonical_move_to_index(m, n, p)
                    assert canonical_index_to_move(idx, n, p) == m
