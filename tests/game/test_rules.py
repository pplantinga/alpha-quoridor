"""Tests for game/rules.py — move generation, wall validation, BFS."""


from game.board import initial_state
from game.rules import apply_move, bfs_path_exists, is_blocked, legal_moves

# ---------------------------------------------------------------------------
# is_blocked
# ---------------------------------------------------------------------------


def test_no_walls_not_blocked() -> None:
    state = initial_state(5, 3)
    # Player 0 at (4,2) — can go N and E and W, not S (board edge)
    assert is_blocked(state, (4, 2), "S")
    assert not is_blocked(state, (4, 2), "N")
    assert not is_blocked(state, (4, 2), "E")
    assert not is_blocked(state, (4, 2), "W")


def test_horizontal_wall_blocks_south() -> None:
    """A horizontal wall at (r,c) should block N↔S movement between rows r and r+1."""
    from game.board import QuoridorState

    state = initial_state(5, 3)
    # Place an 'h' wall at (1, 2) — blocks moving from row 1 to row 2 (south)
    # and from row 2 to row 1 (north) at col 2 and col 3.
    wall_state = QuoridorState(
        board_size=5,
        walls_per_player=3,
        player_pos=state.player_pos,
        walls_remaining=state.walls_remaining,
        placed_walls=frozenset({("h", 1, 2)}),
        current_player=0,
        is_terminal=False,
        winner=None,
    )
    assert is_blocked(wall_state, (1, 2), "S")
    assert is_blocked(wall_state, (2, 2), "N")
    assert is_blocked(wall_state, (1, 3), "S")
    assert is_blocked(wall_state, (2, 3), "N")
    # Unaffected cell
    assert not is_blocked(wall_state, (1, 1), "S")


def test_vertical_wall_blocks_east() -> None:
    """A vertical wall at (r,c) blocks E↔W movement between cols c and c+1."""
    from game.board import QuoridorState

    state = initial_state(5, 3)
    wall_state = QuoridorState(
        board_size=5,
        walls_per_player=3,
        player_pos=state.player_pos,
        walls_remaining=state.walls_remaining,
        placed_walls=frozenset({("v", 1, 2)}),
        current_player=0,
        is_terminal=False,
        winner=None,
    )
    assert is_blocked(wall_state, (1, 2), "E")
    assert is_blocked(wall_state, (1, 3), "W")
    assert is_blocked(wall_state, (2, 2), "E")
    assert is_blocked(wall_state, (2, 3), "W")
    assert not is_blocked(wall_state, (1, 1), "E")


# ---------------------------------------------------------------------------
# Legal pawn moves
# ---------------------------------------------------------------------------


def test_starting_moves_player0() -> None:
    """Player 0 starts at (4,2) on 5x5 — should have 2 pawn moves: N and E/W."""
    state = initial_state(5, 3)
    moves = legal_moves(state)
    pawn_moves = [m for m in moves if m[0] == "move"]
    dests = {(m[1], m[2]) for m in pawn_moves}
    # From (4,2): can go N→(3,2), E→(4,3), W→(4,1). Not S (board edge).
    assert (3, 2) in dests
    assert (4, 3) in dests
    assert (4, 1) in dests
    assert (4, 2) not in dests


def test_straight_jump_over_opponent() -> None:
    """Player can jump straight over the opponent if not blocked."""
    from game.board import QuoridorState

    # Put player 0 at (2,2) and player 1 directly south at (3,2).
    state = QuoridorState(
        board_size=5,
        walls_per_player=0,
        player_pos=((2, 2), (3, 2)),
        walls_remaining=(0, 0),
        placed_walls=frozenset(),
        current_player=0,
        is_terminal=False,
        winner=None,
    )
    moves = legal_moves(state)
    pawn_moves = {(m[1], m[2]) for m in moves if m[0] == "move"}
    assert (4, 2) in pawn_moves  # jumped over opponent
    assert (3, 2) not in pawn_moves  # can't land on opponent


def test_lateral_jump_when_straight_blocked() -> None:
    """When a wall blocks the straight jump, player jumps laterally."""
    from game.board import QuoridorState

    # Player 0 at (2,2), player 1 at (3,2), horizontal wall at (3,2) blocks S.
    state = QuoridorState(
        board_size=5,
        walls_per_player=0,
        player_pos=((2, 2), (3, 2)),
        walls_remaining=(0, 0),
        placed_walls=frozenset({("h", 3, 2)}),
        current_player=0,
        is_terminal=False,
        winner=None,
    )
    moves = legal_moves(state)
    pawn_moves = {(m[1], m[2]) for m in moves if m[0] == "move"}
    # Straight jump blocked → should be able to jump E or W from (3,2)
    assert (3, 3) in pawn_moves or (3, 1) in pawn_moves
    assert (4, 2) not in pawn_moves


# ---------------------------------------------------------------------------
# Wall placement
# ---------------------------------------------------------------------------


def test_wall_count_without_walls() -> None:
    """With 0 walls, no wall moves should be generated."""
    from game.board import QuoridorState

    state = QuoridorState(
        board_size=5,
        walls_per_player=0,
        player_pos=((4, 2), (0, 2)),
        walls_remaining=(0, 0),
        placed_walls=frozenset(),
        current_player=0,
        is_terminal=False,
        winner=None,
    )
    wall_moves = [m for m in legal_moves(state) if m[0] == "wall"]
    assert len(wall_moves) == 0


def test_wall_that_blocks_path_is_illegal() -> None:
    """A wall placement that completely cuts off a player must be rejected."""
    # On a 3x3 board with player 0 at (0,1) and player 1 at (2,1),
    # a horizontal wall at (0,0) and (0,1) would seal the top row.
    from game.board import QuoridorState

    state = QuoridorState(
        board_size=3,
        walls_per_player=5,
        player_pos=((2, 1), (0, 1)),
        walls_remaining=(5, 5),
        placed_walls=frozenset({("h", 0, 0)}),  # blocks (0,*)→(1,*) at col 0&1
        current_player=0,
        is_terminal=False,
        winner=None,
    )
    wall_moves = [m for m in legal_moves(state) if m[0] == "wall"]
    # ("h", 0, 1) would additionally block col 1&2 — player 0 would be sealed at row 1 and 2
    blocking = ("wall", 0, 1, "h")
    assert blocking not in wall_moves


def test_duplicate_wall_illegal() -> None:
    """Placing a wall on an already-occupied slot is illegal."""
    from game.board import QuoridorState

    state = QuoridorState(
        board_size=5,
        walls_per_player=3,
        player_pos=((0, 2), (4, 2)),
        walls_remaining=(3, 3),
        placed_walls=frozenset({("h", 2, 2)}),
        current_player=0,
        is_terminal=False,
        winner=None,
    )
    wall_moves = [m for m in legal_moves(state) if m[0] == "wall"]
    assert ("wall", 2, 2, "h") not in wall_moves


def test_intersecting_wall_illegal() -> None:
    """An 'h' and 'v' wall cannot cross exactly at the same intersection."""
    from game.board import QuoridorState

    state = QuoridorState(
        board_size=5,
        walls_per_player=3,
        player_pos=((0, 2), (4, 2)),
        walls_remaining=(3, 3),
        placed_walls=frozenset({("h", 2, 2)}),
        current_player=0,
        is_terminal=False,
        winner=None,
    )
    wall_moves = [m for m in legal_moves(state) if m[0] == "wall"]
    assert ("wall", 2, 2, "v") not in wall_moves


# ---------------------------------------------------------------------------
# BFS path check
# ---------------------------------------------------------------------------


def test_bfs_open_board() -> None:
    state = initial_state(5, 3)
    assert bfs_path_exists(state, 0)
    assert bfs_path_exists(state, 1)


def test_bfs_already_at_goal() -> None:
    from game.board import QuoridorState

    # Player 0 already at goal row (0 on 5x5)
    state = QuoridorState(
        board_size=5,
        walls_per_player=3,
        player_pos=((0, 2), (4, 2)),
        walls_remaining=(3, 3),
        placed_walls=frozenset(),
        current_player=0,
        is_terminal=True,
        winner=0,
    )
    assert bfs_path_exists(state, 0)


# ---------------------------------------------------------------------------
# apply_move
# ---------------------------------------------------------------------------


def test_apply_pawn_move() -> None:
    state = initial_state(5, 3)
    move = ("move", 3, 2)
    new_state = apply_move(state, move)
    assert new_state.player_pos[0] == (3, 2)
    assert new_state.player_pos[1] == state.player_pos[1]
    assert new_state.current_player == 1
    assert not new_state.is_terminal


def test_apply_wall_move() -> None:
    state = initial_state(5, 3)
    move = ("wall", 1, 1, "h")
    new_state = apply_move(state, move)
    assert ("h", 1, 1) in new_state.placed_walls
    assert new_state.walls_remaining[0] == 2
    assert new_state.walls_remaining[1] == 3
    assert new_state.current_player == 1


def test_win_detection() -> None:
    """Player 0 reaching the goal row (0) should trigger terminal state."""
    from game.board import QuoridorState

    state = QuoridorState(
        board_size=5,
        walls_per_player=3,
        player_pos=((1, 2), (4, 2)),
        walls_remaining=(3, 3),
        placed_walls=frozenset(),
        current_player=0,
        is_terminal=False,
        winner=None,
    )
    new_state = apply_move(state, ("move", 0, 2))
    assert new_state.is_terminal
    assert new_state.winner == 0


def test_legal_moves_empty_when_terminal() -> None:
    state = initial_state(5, 3)
    terminal = apply_move(
        state.__class__(
            board_size=5,
            walls_per_player=3,
            player_pos=((1, 2), (4, 2)),
            walls_remaining=(3, 3),
            placed_walls=frozenset(),
            current_player=0,
            is_terminal=False,
            winner=None,
        ),
        ("move", 0, 2),
    )
    assert terminal.is_terminal
    assert legal_moves(terminal) == []
