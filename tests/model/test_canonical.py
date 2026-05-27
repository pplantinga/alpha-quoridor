import torch

from game.board import QuoridorState
from model.network import encode_state


def test_canonical_symmetry():
    # Board size 9
    n = 9

    # State 1: P0 at row 8, col 4 (Bottom). P1 at row 0, col 4 (Top). CP=0.
    state0 = QuoridorState(
        board_size=n,
        walls_per_player=10,
        player_pos=((8, 4), (0, 4)),
        walls_remaining=(10, 10),
        placed_walls=frozenset(),
        current_player=0,
        is_terminal=False,
        winner=None
    )

    # State 2: P1 at row 0, col 4 (Top). P0 at row 8, col 4 (Bottom). CP=1.
    # Actually, in state1, P0 is at (8,4). In state2, if we want symmetry,
    # the CURRENT player (P1) should be at (0, 4) and opponent (P0) at (8, 4).
    # Wait, my flipping logic: cp=1 flips rows.
    # P1 at (0, 4) -> flipped row = 8-0 = 8. So it appears at (8, 4) in channel 0.
    # P0 at (8, 4) -> flipped row = 8-8 = 0. So it appears at (0, 4) in channel 1.
    # This means state2 (CP=1) should look EXACTLY like state1 (CP=0) in the encoded tensor.
    state1 = QuoridorState(
        board_size=n,
        walls_per_player=10,
        player_pos=((8, 4), (0, 4)),
        walls_remaining=(10, 10),
        placed_walls=frozenset(),
        current_player=1,
        is_terminal=False,
        winner=None
    )

    e0 = encode_state(state0)
    e1 = encode_state(state1)

    # Compare current player positions (Channel 0)
    assert torch.equal(e0[0], e1[0]), "Current player position should be same for both CP=0 and CP=1 in canonical encoding"
    # Compare opponent positions (Channel 1)
    assert torch.equal(e0[1], e1[1]), "Opponent position should be same for both CP=0 and CP=1 in canonical encoding"

    # Test wall flip
    # Horizontal wall at (7, 4) for P0 (blocking 7-8)
    state2 = QuoridorState(
        board_size=n,
        walls_per_player=10,
        player_pos=((8, 4), (0, 4)),
        walls_remaining=(10, 10),
        placed_walls=frozenset([("h", 7, 4)]),
        current_player=0,
        is_terminal=False,
        winner=None
    )

    # Flipped symmetric wall for P1: Horizontal wall at row (9-2-7) = 0.
    # Blocking between row 0 and 1.
    state3 = QuoridorState(
        board_size=n,
        walls_per_player=10,
        player_pos=((8, 4), (0, 4)),
        walls_remaining=(10, 10),
        placed_walls=frozenset([("h", 0, 4)]),
        current_player=1,
        is_terminal=False,
        winner=None
    )

    e2 = encode_state(state2)
    e3 = encode_state(state3)

    assert torch.equal(e2[2], e3[2]), "Horizontal wall should be symmetric after flipping"

    print("Canonical symmetry tests passed!")

if __name__ == "__main__":
    test_canonical_symmetry()
