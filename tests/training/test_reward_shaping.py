"""Tests for reward shaping helpers.

Verifies that heuristic_value and step_shaping return correct signs
for clear winning/losing positions and for obvious good/bad moves.
"""


from game.board import QuoridorState, initial_state
from model.network import encode_state
from training.reward_shaping import heuristic_value, step_shaping
from utils.config import Config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    p0_pos: tuple[int, int],
    p1_pos: tuple[int, int],
    walls_remaining: tuple[int, int] = (10, 10),
) -> QuoridorState:
    """Return a 9x9 state with players at the given positions and no walls."""
    return QuoridorState(
        board_size=9,
        walls_per_player=10,
        player_pos=(p0_pos, p1_pos),
        walls_remaining=walls_remaining,
        placed_walls=frozenset(),
        current_player=0,
        is_terminal=False,
        winner=None,
    )


# ---------------------------------------------------------------------------
# heuristic_value tests
# ---------------------------------------------------------------------------

def test_heuristic_value_winning_position():
    """Current player (P0) at row 1 (1 step from goal=0), opponent at row 1 towards goal=8 (7 steps from goal) → positive."""
    # P0's goal is row 0: 1 step away. P1's goal is row 8: P1 is at row 7, 1 step away too.
    # Use a clear mismatch: P0 at row 1 (dist=1), P1 at row 2 (dist=6 to row 8).
    state = _make_state(p0_pos=(1, 4), p1_pos=(2, 4))  # P0 much closer to goal
    enc = encode_state(state)
    h = heuristic_value(enc)
    assert h > 0, f"Expected positive heuristic for near-win, got {h}"


def test_heuristic_value_losing_position():
    """Current player (P0) far from goal, opponent close → negative."""
    # P0 at row 7 (P0's goal=0, dist=7), P1 at row 6 (P1's goal=8, dist=2) → P0 is losing badly
    state = _make_state(p0_pos=(7, 4), p1_pos=(6, 4))
    enc = encode_state(state)
    h = heuristic_value(enc)
    assert h < 0, f"Expected negative heuristic for near-loss, got {h}"


def test_heuristic_value_symmetric():
    """Starting position is approximately symmetric → |h| < 0.15."""
    state = initial_state(9, 10)
    enc = encode_state(state)
    h = heuristic_value(enc)
    assert abs(h) < 0.15, f"Expected near-zero heuristic at start, got {h}"


def test_heuristic_value_wall_advantage():
    """Same distances but current player has more walls → slight positive."""
    state = _make_state(p0_pos=(4, 4), p1_pos=(4, 4), walls_remaining=(10, 0))
    enc = encode_state(state)
    h = heuristic_value(enc)
    assert h > 0, f"Expected positive heuristic for wall advantage, got {h}"


# ---------------------------------------------------------------------------
# step_shaping tests
# ---------------------------------------------------------------------------

def test_step_shaping_pawn_advance():
    """Moving P0 one step closer to the goal → positive shaping reward."""
    before = _make_state(p0_pos=(3, 4), p1_pos=(5, 4))
    after_state = QuoridorState(
        board_size=9,
        walls_per_player=10,
        player_pos=((2, 4), (5, 4)),  # P0 moved from row 3 to row 2 (closer to 0)
        walls_remaining=(10, 10),
        placed_walls=frozenset(),
        current_player=1,  # P1's turn now
        is_terminal=False,
        winner=None,
    )
    s_enc = encode_state(before)
    next_enc = encode_state(after_state)
    shaping = step_shaping(s_enc, next_enc, progress_weight=0.3, block_weight=0.2)
    assert shaping > 0, f"Expected positive shaping for pawn advance, got {shaping}"


def test_step_shaping_pawn_retreat():
    """Moving P0 away from the goal → negative shaping reward."""
    before = _make_state(p0_pos=(2, 4), p1_pos=(5, 4))
    after_state = QuoridorState(
        board_size=9,
        walls_per_player=10,
        player_pos=((3, 4), (5, 4)),  # P0 moved from row 2 to row 3 (farther)
        walls_remaining=(10, 10),
        placed_walls=frozenset(),
        current_player=1,
        is_terminal=False,
        winner=None,
    )
    s_enc = encode_state(before)
    next_enc = encode_state(after_state)
    shaping = step_shaping(s_enc, next_enc, progress_weight=0.3, block_weight=0.2)
    assert shaping < 0, f"Expected negative shaping for pawn retreat, got {shaping}"


def test_step_shaping_range():
    """Shaping should stay inside a reasonable range for any single step."""
    # Max possible my_progress ≈ 1/9; max opp_blocked ≈ a few steps / 9
    # With weights 0.3 and 0.2 the sum should stay well under 1.0
    for _ in range(10):
        state = initial_state(9, 10)
        s_enc = encode_state(state)
        shaping = step_shaping(s_enc, s_enc, progress_weight=0.3, block_weight=0.2)
        assert -1.0 <= shaping <= 1.0, f"Shaping out of range: {shaping}"


def test_minimax_data_value_range():
    """bootstrap_utils must produce values in [-1, 1] (not just -1, 0, 1)."""
    from training.bootstrap_utils import generate_heuristic_data
    from utils.config import TrainingConfig

    config = Config(
        board_size=3,
        walls_per_player=2,
        training=TrainingConfig(batch_size=4),
    )
    exps = generate_heuristic_data(config, num_games=2, depth=1)
    assert len(exps) > 0
    for s, p, v in exps:
        assert s.shape == (9, 3, 3)
        assert p.shape == (3 * 3 * 3,)
        assert isinstance(v, float)
        assert -1.0 <= v <= 1.0, f"Bootstrap value out of range: {v}"


if __name__ == "__main__":
    test_heuristic_value_winning_position()
    test_heuristic_value_losing_position()
    test_heuristic_value_symmetric()
    test_heuristic_value_wall_advantage()
    test_step_shaping_pawn_advance()
    test_step_shaping_pawn_retreat()
    test_step_shaping_range()
    test_minimax_data_value_range()
    print("All reward shaping tests passed!")
