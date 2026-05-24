"""Tests for game/board.py — state representation and initial_state."""

from game.board import goal_row, initial_state


def test_initial_state_9x9() -> None:
    state = initial_state(9, 10)
    assert state.board_size == 9
    assert state.walls_per_player == 10
    assert state.player_pos[0] == (8, 4)
    assert state.player_pos[1] == (0, 4)
    assert state.walls_remaining == (10, 10)
    assert len(state.placed_walls) == 0
    assert state.current_player == 0
    assert not state.is_terminal
    assert state.winner is None


def test_initial_state_5x5() -> None:
    state = initial_state(5, 3)
    assert state.player_pos[0] == (4, 2)
    assert state.player_pos[1] == (0, 2)
    assert state.walls_remaining == (3, 3)


def test_initial_state_3x3() -> None:
    state = initial_state(3, 1)
    assert state.player_pos[0] == (2, 1)
    assert state.player_pos[1] == (0, 1)


def test_goal_rows() -> None:
    assert goal_row(0, 9) == 0
    assert goal_row(1, 9) == 8
    assert goal_row(0, 5) == 0
    assert goal_row(1, 5) == 4


def test_state_is_frozen() -> None:
    """QuoridorState must be immutable (frozen dataclass)."""
    state = initial_state(9, 10)
    try:
        setattr(state, "current_player", 1)
        assert False, "Should have raised FrozenInstanceError"
    except Exception:
        pass
