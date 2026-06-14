"""Tests for left-right symmetry augmentation."""

import torch
import pytest

from game.board import QuoridorState
from model.network import canonical_move_to_index, encode_state
from training.augmentation import flip_experience, flip_policy, flip_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    n: int = 9,
    cp: int = 0,
    pos0: tuple[int, int] = (8, 4),
    pos1: tuple[int, int] = (0, 4),
    walls: frozenset = frozenset(),
    walls_remaining: tuple[int, int] = (10, 10),
) -> QuoridorState:
    return QuoridorState(
        board_size=n,
        walls_per_player=10,
        player_pos=(pos0, pos1),
        walls_remaining=walls_remaining,
        placed_walls=walls,
        current_player=cp,
        is_terminal=False,
        winner=None,
    )


# ---------------------------------------------------------------------------
# State flip tests
# ---------------------------------------------------------------------------

class TestFlipState:
    def test_pawn_position_mirrors_correctly(self):
        """A pawn at column c should appear at column n-1-c after the flip."""
        n = 9
        state = _make_state(n=n, pos0=(4, 2), pos1=(4, 6))
        enc = encode_state(state)
        flipped = flip_state(enc)

        # Current-player pawn was at col 2 → should now be at col 6
        assert flipped[0, 4, 6] == pytest.approx(1.0)
        assert flipped[0].sum() == pytest.approx(1.0)

        # Opponent pawn was at col 6 → should now be at col 2
        assert flipped[1, 4, 2] == pytest.approx(1.0)
        assert flipped[1].sum() == pytest.approx(1.0)

    def test_flip_is_involution(self):
        """Flipping twice should return the original tensor."""
        state = _make_state(pos0=(7, 1), pos1=(2, 7))
        enc = encode_state(state)
        assert torch.allclose(flip_state(flip_state(enc)), enc)

    def test_scalar_channels_unchanged(self):
        """Channels 4, 5, 6 (scalar broadcasts) must be unchanged by a flip."""
        state = _make_state(walls_remaining=(7, 4))
        enc = encode_state(state)
        flipped = flip_state(enc)
        for ch in (4, 5, 6):
            assert torch.allclose(flipped[ch], enc[ch]), f"Channel {ch} changed after flip"

    def test_horizontal_wall_mirrors_correctly(self):
        """An h-wall at (r, c) should map to (r, n-2-c) after flip."""
        n = 9
        # h-wall at (3, 2): spans cols 2 and 3 → mirror spans cols 5 and 6,
        # so mirrored anchor is col 6 = n-2-c = 9-2-2 = 5... wait:
        # n-2-c = 9-2-2 = 5, so the anchor moves to col 5.
        state = _make_state(n=n, walls=frozenset([("h", 3, 2)]))
        enc = encode_state(state)
        flipped = flip_state(enc)

        # Wall channel 2 (h-walls): original col 2 → mirrored col n-2-2 = 5
        assert flipped[2, 3, 5] == pytest.approx(1.0), (
            f"Expected h-wall at (3,5) but got:\n{flipped[2]}"
        )
        assert flipped[2].sum() == pytest.approx(1.0)

    def test_vertical_wall_mirrors_correctly(self):
        """A v-wall at (r, c) should map to (r, n-2-c) after flip."""
        n = 9
        # v-wall at (2, 1): spans cols 1 and 2 → mirror spans cols 6 and 7,
        # so mirrored anchor = n-2-1 = 6.
        state = _make_state(n=n, walls=frozenset([("v", 2, 1)]))
        enc = encode_state(state)
        flipped = flip_state(enc)

        assert flipped[3, 2, 6] == pytest.approx(1.0), (
            f"Expected v-wall at (2,6) but got:\n{flipped[3]}"
        )
        assert flipped[3].sum() == pytest.approx(1.0)

    def test_wall_column_at_boundary(self):
        """Walls at the last valid column (n-2) should map to column 0."""
        n = 9
        state = _make_state(n=n, walls=frozenset([("h", 0, n - 2)]))
        enc = encode_state(state)
        flipped = flip_state(enc)

        # n-2-c = 9-2-(9-2) = 0
        assert flipped[2, 0, 0] == pytest.approx(1.0)

    def test_distance_map_mirrors_correctly(self):
        """Distance maps should flip columns symmetrically."""
        state = _make_state(pos0=(4, 3), pos1=(4, 5))
        enc = encode_state(state)
        flipped = flip_state(enc)

        for ch in (7, 8):
            assert torch.allclose(flipped[ch], enc[ch].flip(dims=[1])), (
                f"Distance channel {ch} not correctly mirrored"
            )


# ---------------------------------------------------------------------------
# Policy flip tests
# ---------------------------------------------------------------------------

class TestFlipPolicy:
    def test_pawn_move_mirrors_correctly(self):
        """A pawn-move policy spike at (r, c) should map to (r, n-1-c)."""
        n = 9
        policy = torch.zeros(3 * n * n)
        # Pawn move at (4, 2): flat index = 0*N*N + 4*N + 2
        idx_orig = 0 * n * n + 4 * n + 2
        policy[idx_orig] = 1.0

        flipped = flip_policy(policy, n)

        # Should appear at (4, n-1-2) = (4, 6)
        idx_mirror = 0 * n * n + 4 * n + 6
        assert flipped[idx_mirror] == pytest.approx(1.0)
        assert flipped.sum() == pytest.approx(1.0)

    def test_hwall_mirrors_correctly(self):
        """An h-wall policy spike at (r, c) should map to (r, n-2-c)."""
        n = 9
        policy = torch.zeros(3 * n * n)
        r, c = 3, 2
        idx_orig = 1 * n * n + r * n + c
        policy[idx_orig] = 1.0

        flipped = flip_policy(policy, n)

        # n-2-c = 9-2-2 = 5
        idx_mirror = 1 * n * n + r * n + (n - 2 - c)
        assert flipped[idx_mirror] == pytest.approx(1.0), (
            f"Expected h-wall at col {n-2-c}, got distribution:\n"
            f"{flipped[n*n:2*n*n].view(n,n)}"
        )
        assert flipped.sum() == pytest.approx(1.0)

    def test_vwall_mirrors_correctly(self):
        """A v-wall policy spike at (r, c) should map to (r, n-2-c)."""
        n = 9
        policy = torch.zeros(3 * n * n)
        r, c = 2, 1
        idx_orig = 2 * n * n + r * n + c
        policy[idx_orig] = 1.0

        flipped = flip_policy(policy, n)

        # n-2-c = 9-2-1 = 6
        idx_mirror = 2 * n * n + r * n + (n - 2 - c)
        assert flipped[idx_mirror] == pytest.approx(1.0)
        assert flipped.sum() == pytest.approx(1.0)

    def test_policy_flip_is_involution(self):
        """Flipping a policy vector twice should return the original."""
        n = 9
        policy = torch.rand(3 * n * n)
        # Zero out always-invalid wall positions (col n-1) to avoid ambiguity
        for ch in (1, 2):
            for r in range(n):
                policy[ch * n * n + r * n + (n - 1)] = 0.0
        assert torch.allclose(flip_policy(flip_policy(policy, n), n), policy)

    def test_probability_mass_preserved(self):
        """Flipping should preserve the total probability mass.

        Wall channels never assign probability to column n-1 (no wall can start
        at the last column), so we zero those slots before flipping to match the
        contract of real MCTS-derived policy vectors.
        """
        n = 9
        policy = torch.softmax(torch.randn(3 * n * n), dim=0)
        # Zero out always-invalid wall positions (col n-1 in each wall channel)
        for ch in (1, 2):
            for r in range(n):
                policy[ch * n * n + r * n + (n - 1)] = 0.0
        flipped = flip_policy(policy, n)
        assert flipped.sum() == pytest.approx(policy.sum().item(), abs=1e-5)


# ---------------------------------------------------------------------------
# Round-trip consistency test
# ---------------------------------------------------------------------------

class TestFlipExperience:
    def test_value_unchanged(self):
        """Value should be identical in the mirrored experience."""
        state = _make_state()
        enc = encode_state(state)
        policy = torch.zeros(3 * 9 * 9)
        value = 0.73

        _, _, v_flip = flip_experience(enc, policy, value, 9)
        assert v_flip == pytest.approx(value)

    def test_encoded_state_consistency(self):
        """
        Encoding a physically mirrored state should match flip_state applied to
        the original encoding, for pawn-only positions (no walls, for simplicity).

        We mirror the state by swapping col c → n-1-c for both pawns.
        """
        n = 9
        c0, c1 = 2, 6
        state = _make_state(n=n, pos0=(7, c0), pos1=(1, c1))
        enc = encode_state(state)
        enc_flipped = flip_state(enc)

        # Build the physically mirrored state
        mirrored_state = _make_state(n=n, pos0=(7, n - 1 - c0), pos1=(1, n - 1 - c1))
        enc_mirrored = encode_state(mirrored_state)

        # Pawn channels (0, 1) and distance maps (7, 8) should match exactly
        for ch in (0, 1, 7, 8):
            assert torch.allclose(enc_flipped[ch], enc_mirrored[ch], atol=1e-5), (
                f"Channel {ch} mismatch between flip_state and physically mirrored encoding"
            )
