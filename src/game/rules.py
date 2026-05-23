"""Move generation, validation, and game rules for Quoridor."""

from __future__ import annotations

import heapq
from collections import deque
from functools import lru_cache

from game.board import Move, QuoridorState, Wall, goal_row, initial_state  # noqa: F401

# Directional deltas: (dr, dc)
_DIRECTIONS: dict[str, tuple[int, int]] = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
}


# ---------------------------------------------------------------------------
# Wall blocking helpers
# ---------------------------------------------------------------------------


def _wall_blocks_h(walls: frozenset[Wall], r: int, c: int, extra_wall: Wall | None = None) -> bool:
    """Return True if a horizontal wall blocks moving from (r, c) to (r+1, c)."""
    if extra_wall == ("h", r, c) or extra_wall == ("h", r, c - 1):
        return True
    return ("h", r, c) in walls or ("h", r, c - 1) in walls


def _wall_blocks_v(walls: frozenset[Wall], r: int, c: int, extra_wall: Wall | None = None) -> bool:
    """Return True if a vertical wall blocks moving from (r, c) to (r, c+1)."""
    if extra_wall == ("v", r, c) or extra_wall == ("v", r - 1, c):
        return True
    return ("v", r, c) in walls or ("v", r - 1, c) in walls


def is_blocked(
    state: QuoridorState,
    pos: tuple[int, int],
    direction: str,
    extra_wall: Wall | None = None,
) -> bool:
    """Return True if movement from pos in direction is blocked by a wall."""
    r, c = pos
    walls = state.placed_walls

    if direction == "N":
        return r == 0 or _wall_blocks_h(walls, r - 1, c, extra_wall)
    if direction == "S":
        return r == state.board_size - 1 or _wall_blocks_h(walls, r, c, extra_wall)
    if direction == "E":
        return c == state.board_size - 1 or _wall_blocks_v(walls, r, c, extra_wall)
    if direction == "W":
        return c == 0 or _wall_blocks_v(walls, r, c - 1, extra_wall)
    raise ValueError(f"Unknown direction: {direction}")


def _is_on_board(board_size: int, r: int, c: int) -> bool:
    return 0 <= r < board_size and 0 <= c < board_size


# ---------------------------------------------------------------------------
# Pawn move generation (including jump rules)
# ---------------------------------------------------------------------------


def _pawn_destinations(state: QuoridorState, player: int) -> list[tuple[int, int]]:
    """Return all squares the given player's pawn can legally move to."""
    pos = state.player_pos[player]
    opp_pos = state.player_pos[1 - player]
    n = state.board_size
    destinations: list[tuple[int, int]] = []

    for d, (dr, dc) in _DIRECTIONS.items():
        if is_blocked(state, pos, d):
            continue
        nr, nc = pos[0] + dr, pos[1] + dc
        if (nr, nc) == opp_pos:
            # Jump logic: try straight jump first
            jump_d = d
            if not is_blocked(state, opp_pos, jump_d):
                jr, jc = nr + dr, nc + dc
                destinations.append((jr, jc))
            else:
                # Blocked straight — try lateral jumps
                for ld, (ldr, ldc) in _DIRECTIONS.items():
                    if ld == d or ld == _opposite(d):
                        continue
                    if not is_blocked(state, opp_pos, ld):
                        lr, lc = nr + ldr, nc + ldc
                        if _is_on_board(n, lr, lc):
                            destinations.append((lr, lc))
        else:
            destinations.append((nr, nc))

    return destinations


def _opposite(direction: str) -> str:
    return {"N": "S", "S": "N", "E": "W", "W": "E"}[direction]


# ---------------------------------------------------------------------------
# BFS path check
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1024)
def a_star_path(
    state: QuoridorState, player: int, extra_wall: Wall | None = None
) -> list[tuple[int, int]] | None:
    """Return an A* shortest path to the goal row, or None if blocked."""
    target_row = goal_row(player, state.board_size)
    start = state.player_pos[player]

    if start[0] == target_row:
        return [start]

    # Priority queue: (f_score, g_score, (r, c))
    frontier: list[tuple[int, int, tuple[int, int]]] = []
    heapq.heappush(frontier, (abs(start[0] - target_row), 0, start))

    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], int] = {start: 0}

    while frontier:
        _, current_g, current = heapq.heappop(frontier)

        if current[0] == target_row:
            path = []
            curr: tuple[int, int] | None = current
            while curr is not None:
                path.append(curr)
                curr = came_from[curr]
            path.reverse()
            return path

        if current_g > g_score.get(current, float("inf")):
            continue

        for d, (dr, dc) in _DIRECTIONS.items():
            if is_blocked(state, current, d, extra_wall=extra_wall):
                continue
            nr, nc = current[0] + dr, current[1] + dc
            next_node = (nr, nc)
            new_g = current_g + 1

            if new_g < g_score.get(next_node, float("inf")):
                came_from[next_node] = current
                g_score[next_node] = new_g
                priority = new_g + abs(nr - target_row)
                heapq.heappush(frontier, (priority, new_g, next_node))

    return None


try:
    from game.rules_fast import has_path_to_goal_cython
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False


def has_path_to_goal(state: QuoridorState, player: int, extra_wall: Wall | None = None) -> bool:
    """Return True if the player has a path to their goal row."""
    if _HAS_CYTHON:
        return has_path_to_goal_cython(
            state.board_size,
            state.player_pos[player],
            goal_row(player, state.board_size),
            state.placed_walls,
            extra_wall
        )

    target_row = goal_row(player, state.board_size)
    start = state.player_pos[player]

    if start[0] == target_row:
        return True

    queue = deque([start])
    visited = {start}

    while queue:
        current = queue.popleft()
        if current[0] == target_row:
            return True

        for d, (dr, dc) in _DIRECTIONS.items():
            if is_blocked(state, current, d, extra_wall=extra_wall):
                continue
            nr, nc = current[0] + dr, current[1] + dc
            next_node = (nr, nc)
            if next_node not in visited:
                visited.add(next_node)
                queue.append(next_node)

    return False


# Alias for backward compatibility
bfs_path_exists = has_path_to_goal


def _path_intersects_wall(
    path: list[tuple[int, int]], r: int, c: int, orient: str
) -> bool:
    """Return True if the specified wall directly blocks any step in the path."""
    if not path:
        return False

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]

        if orient == "h":
            # Horizontal wall blocks vertical movement between row r and r+1 at col c or c+1
            if min(r1, r2) == r and max(r1, r2) == r + 1:
                if c1 == c2 and (c1 == c or c1 == c + 1):
                    return True
        else:
            # Vertical wall blocks horizontal movement between col c and c+1 at row r or r+1
            if min(c1, c2) == c and max(c1, c2) == c + 1:
                if r1 == r2 and (r1 == r or r1 == r + 1):
                    return True

    return False


# ---------------------------------------------------------------------------
# Wall placement validation
# ---------------------------------------------------------------------------


def _wall_overlaps(walls: frozenset[Wall], wall: Wall) -> bool:
    """Return True if the wall overlaps or crosses an existing wall."""
    orient, r, c = wall
    if wall in walls:
        return True
    if orient == "h":
        # Overlaps with adjacent horizontal wall sharing a cell
        if ("h", r, c - 1) in walls or ("h", r, c + 1) in walls:
            return True
        # Crosses a vertical wall at the exact same intersection
        if ("v", r, c) in walls:
            return True
    else:  # vertical
        if ("v", r - 1, c) in walls or ("v", r + 1, c) in walls:
            return True
        if ("h", r, c) in walls:
            return True
    return False


def _wall_in_bounds(board_size: int, wall: Wall) -> bool:
    """Return True if the wall fits within the board."""
    orient, r, c = wall
    n = board_size
    if orient == "h":
        # Blocks between rows r and r+1 at cols c and c+1
        return 0 <= r < n - 1 and 0 <= c < n - 1
    else:
        # Blocks between cols c and c+1 at rows r and r+1
        return 0 <= r < n - 1 and 0 <= c < n - 1


# ---------------------------------------------------------------------------
# Legal move generation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1024)
def legal_moves(state: QuoridorState) -> list[Move]:
    """Return all legal moves for the current player."""
    if state.is_terminal:
        return []

    player = state.current_player
    moves: list[Move] = []

    # Pawn moves
    for dest in _pawn_destinations(state, player):
        moves.append(("move", dest[0], dest[1]))

    # Wall placements
    if state.walls_remaining[player] > 0:
        n = state.board_size

        # Precompute the shortest path for both players
        p0_path = a_star_path(state, 0)
        p1_path = a_star_path(state, 1)

        for r in range(n - 1):
            for c in range(n - 1):
                for orient in ("h", "v"):
                    wall: Wall = (orient, r, c)
                    if not _wall_in_bounds(n, wall) or _wall_overlaps(
                        state.placed_walls, wall
                    ):
                        continue

                    blocks_p0 = p0_path is not None and _path_intersects_wall(
                        p0_path, r, c, orient
                    )
                    blocks_p1 = p1_path is not None and _path_intersects_wall(
                        p1_path, r, c, orient
                    )

                    is_valid = True
                    if blocks_p0 or blocks_p1:
                        # Fallback: run fast BFS treating the new wall as placed
                        if blocks_p0 and not has_path_to_goal(state, 0, extra_wall=wall):
                            is_valid = False
                        if is_valid and blocks_p1 and not has_path_to_goal(state, 1, extra_wall=wall):
                            is_valid = False

                    if is_valid:
                        moves.append(("wall", r, c, orient))

    return moves


# ---------------------------------------------------------------------------
# Apply move
# ---------------------------------------------------------------------------


def apply_move(state: QuoridorState, move: Move) -> QuoridorState:
    """Return the new state after the current player applies move."""
    player = state.current_player
    next_player = 1 - player

    if move[0] == "move":
        _, nr, nc = move
        new_pos = list(state.player_pos)
        new_pos[player] = (nr, nc)
        pos_tuple = (new_pos[0], new_pos[1])

        target = goal_row(player, state.board_size)
        terminal = nr == target
        winner = player if terminal else None

        return QuoridorState(
            board_size=state.board_size,
            walls_per_player=state.walls_per_player,
            player_pos=pos_tuple,
            walls_remaining=state.walls_remaining,
            placed_walls=state.placed_walls,
            current_player=next_player,
            is_terminal=terminal,
            winner=winner,
        )

    elif move[0] == "wall":
        _, r, c, orient = move
        wall: Wall = (orient, r, c)
        new_walls = state.placed_walls | {wall}
        new_remaining = list(state.walls_remaining)
        new_remaining[player] -= 1

        return QuoridorState(
            board_size=state.board_size,
            walls_per_player=state.walls_per_player,
            player_pos=state.player_pos,
            walls_remaining=(new_remaining[0], new_remaining[1]),
            placed_walls=new_walls,
            current_player=next_player,
            is_terminal=False,
            winner=None,
        )

    raise ValueError(f"Unknown move type: {move[0]}")
