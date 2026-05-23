# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

"""Cython implementation of Quoridor rules engine."""

from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

# Using a fixed size for the stack-based queue to avoid allocations
# For 9x9 board, 81 nodes is max.
DEF MAX_NODES = 256 

cdef struct Coord:
    int r
    int c

def has_path_to_goal_cython(int board_size, 
                            tuple player_pos, 
                            int target_row, 
                            object placed_walls, 
                            object extra_wall=None):
    """Fast C-level reachability check."""
    cdef int n = board_size
    cdef int start_r = player_pos[0]
    cdef int start_c = player_pos[1]
    
    if start_r == target_row:
        return True
        
    # Convert walls to 2D C-arrays for O(1) access
    cdef char h_walls[10][10]
    cdef char v_walls[10][10]
    cdef int i, j
    
    for i in range(10):
        for j in range(10):
            h_walls[i][j] = 0
            v_walls[i][j] = 0
            
    for w in placed_walls:
        if w[0] == 'h':
            h_walls[w[1]][w[2]] = 1
        else:
            v_walls[w[1]][w[2]] = 1
            
    if extra_wall is not None:
        if extra_wall[0] == 'h':
            h_walls[extra_wall[1]][extra_wall[2]] = 1
        else:
            v_walls[extra_wall[1]][extra_wall[2]] = 1

    # BFS using a simple queue
    cdef Coord queue[MAX_NODES]
    cdef int head = 0
    cdef int tail = 0
    
    cdef char visited[10][10]
    for i in range(10):
        for j in range(10):
            visited[i][j] = 0
            
    queue[tail].r = start_r
    queue[tail].c = start_c
    tail += 1
    visited[start_r][start_c] = 1
    
    cdef int r, c, nr, nc
    cdef Coord curr
    
    while head < tail:
        curr = queue[head]
        head += 1
        r = curr.r
        c = curr.c
        
        if r == target_row:
            return True
            
        # North
        if r > 0:
            if not (h_walls[r-1][c] or (c > 0 and h_walls[r-1][c-1])):
                nr, nc = r - 1, c
                if not visited[nr][nc]:
                    visited[nr][nc] = 1
                    queue[tail].r = nr
                    queue[tail].c = nc
                    tail += 1
        
        # South
        if r < n - 1:
            if not (h_walls[r][c] or (c > 0 and h_walls[r][c-1])):
                nr, nc = r + 1, c
                if not visited[nr][nc]:
                    visited[nr][nc] = 1
                    queue[tail].r = nr
                    queue[tail].c = nc
                    tail += 1
                    
        # East
        if c < n - 1:
            if not (v_walls[r][c] or (r > 0 and v_walls[r-1][c])):
                nr, nc = r, c + 1
                if not visited[nr][nc]:
                    visited[nr][nc] = 1
                    queue[tail].r = nr
                    queue[tail].c = nc
                    tail += 1
                    
        # West
        if c > 0:
            if not (v_walls[r][c-1] or (r > 0 and v_walls[r-1][c-1])):
                nr, nc = r, c - 1
                if not visited[nr][nc]:
                    visited[nr][nc] = 1
                    queue[tail].r = nr
                    queue[tail].c = nc
                    tail += 1
                    
    return False
