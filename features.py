"""
Numpy-optimized game analysis utilities for Tron.
Part of Workflow A - Core Engine Implementation.

All functions are deterministic and use BFS with consistent neighbor ordering.
"""

import numpy as np
from collections import deque


# Deterministic neighbor order: Up, Right, Down, Left
NEIGHBORS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def reachable_area(grid, head):
    """
    Count number of reachable empty cells from a head position using BFS.
    
    Args:
        grid: 2D numpy array (0=empty, non-zero=occupied)
        head: Tuple (row, col) starting position
        
    Returns:
        Integer count of reachable empty cells (including starting position if empty)
    """
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    
    # Treat the head cell as empty for purposes of calculating reachable area
    # This ensures the heuristic bot won't produce degenerate evaluations
    # when evaluating moves from positions that are already marked on the grid
    temp_grid = grid.copy()
    if temp_grid[head] != 0:
        temp_grid[head] = 0
    
    queue = deque([head])
    visited[head] = True
    count = 1
    
    while queue:
        row, col = queue.popleft()
        
        # Process neighbors in deterministic order: Up, Right, Down, Left
        for dr, dc in NEIGHBORS:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if new_row < 0 or new_row >= rows or new_col < 0 or new_col >= cols:
                continue
            
            # Check if already visited
            if visited[new_row, new_col]:
                continue
            
            # Check if empty (using temp_grid where head is treated as empty)
            if temp_grid[new_row, new_col] != 0:
                continue
            
            # Mark as visited and add to queue
            visited[new_row, new_col] = True
            queue.append((new_row, new_col))
            count += 1
    
    return count


def voronoi_counts(grid, head_me, head_opp):
    """
    Multi-source BFS to compute Voronoi regions.
    
    Cells closer to me -> me
    Cells closer to opp -> opp
    Equal distance -> tie
    
    Args:
        grid: 2D numpy array (0=empty, non-zero=occupied)
        head_me: Tuple (row, col) for my head
        head_opp: Tuple (row, col) for opponent's head
        
    Returns:
        Tuple (me_count, opp_count, tie_count)
    """
    rows, cols = grid.shape
    
    # Distance arrays (-1 = unvisited)
    dist_me = np.full((rows, cols), -1, dtype=np.int32)
    dist_opp = np.full((rows, cols), -1, dtype=np.int32)
    
    # Initialize BFS queues
    queue_me = deque([head_me])
    queue_opp = deque([head_opp])
    
    # Mark starting positions (distance 0)
    if grid[head_me] == 0:
        dist_me[head_me] = 0
    if grid[head_opp] == 0:
        dist_opp[head_opp] = 0
    
    # BFS from my head
    while queue_me:
        row, col = queue_me.popleft()
        current_dist = dist_me[row, col]
        
        for dr, dc in NEIGHBORS:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if new_row < 0 or new_row >= rows or new_col < 0 or new_col >= cols:
                continue
            
            # Check if already visited
            if dist_me[new_row, new_col] != -1:
                continue
            
            # Check if empty
            if grid[new_row, new_col] != 0:
                continue
            
            # Update distance and add to queue
            dist_me[new_row, new_col] = current_dist + 1
            queue_me.append((new_row, new_col))
    
    # BFS from opponent's head
    while queue_opp:
        row, col = queue_opp.popleft()
        current_dist = dist_opp[row, col]
        
        for dr, dc in NEIGHBORS:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if new_row < 0 or new_row >= rows or new_col < 0 or new_col >= cols:
                continue
            
            # Check if already visited
            if dist_opp[new_row, new_col] != -1:
                continue
            
            # Check if empty
            if grid[new_row, new_col] != 0:
                continue
            
            # Update distance and add to queue
            dist_opp[new_row, new_col] = current_dist + 1
            queue_opp.append((new_row, new_col))
    
    # Count cells in each category
    me_count = 0
    opp_count = 0
    tie_count = 0
    
    for row in range(rows):
        for col in range(cols):
            d_me = dist_me[row, col]
            d_opp = dist_opp[row, col]
            
            # Skip if unreachable by both
            if d_me == -1 and d_opp == -1:
                continue
            
            # Only reachable by me
            if d_me != -1 and d_opp == -1:
                me_count += 1
            # Only reachable by opponent
            elif d_me == -1 and d_opp != -1:
                opp_count += 1
            # Reachable by both
            else:
                if d_me < d_opp:
                    me_count += 1
                elif d_opp < d_me:
                    opp_count += 1
                else:
                    tie_count += 1
    
    return me_count, opp_count, tie_count


def is_tunnel(grid, head):
    """
    Check if a position is a tunnel.
    
    A position is considered a tunnel if:
    - It has ≤1 free neighbor; OR
    - It is entering a 1-wide corridor (short forward probe)
    
    Args:
        grid: 2D numpy array (0=empty, non-zero=occupied)
        head: Tuple (row, col) position to check
        
    Returns:
        Boolean indicating if position is a tunnel
    """
    rows, cols = grid.shape
    row, col = head
    
    # Check bounds
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return True
    
    # If position is occupied, it's not a valid tunnel check
    if grid[row, col] != 0:
        return True
    
    # Count free neighbors
    free_neighbors = []
    for dr, dc in NEIGHBORS:
        new_row, new_col = row + dr, col + dc
        
        # Check bounds
        if new_row < 0 or new_row >= rows or new_col < 0 or new_col >= cols:
            continue
        
        # Check if empty
        if grid[new_row, new_col] == 0:
            free_neighbors.append((new_row, new_col))
    
    # If ≤1 free neighbor, it's a tunnel
    if len(free_neighbors) <= 1:
        return True
    
    # Short forward probe: check if entering a 1-wide corridor
    # For each free neighbor, check if it also has ≤1 free neighbor (excluding current)
    for neighbor in free_neighbors:
        n_row, n_col = neighbor
        n_free_count = 0
        
        for dr, dc in NEIGHBORS:
            new_row, new_col = n_row + dr, n_col + dc
            
            # Check bounds
            if new_row < 0 or new_row >= rows or new_col < 0 or new_col >= cols:
                continue
            
            # Skip current position
            if (new_row, new_col) == (row, col):
                continue
            
            # Check if empty
            if grid[new_row, new_col] == 0:
                n_free_count += 1
        
        # If neighbor has ≤1 other free neighbor, we're entering a corridor
        if n_free_count <= 1:
            return True
    
    return False


def simulate_step(grid, head, action):
    """
    Simulate one step without modifying the input grid.
    
    Args:
        grid: 2D numpy array (0=empty, non-zero=occupied)
        head: Tuple (row, col) current position
        action: Action index (0=Up, 1=Right, 2=Down, 3=Left)
        
    Returns:
        Tuple (new_head, crashed)
        - new_head: Tuple (row, col) of new position
        - crashed: Boolean indicating if move results in crash
    """
    # Direction vectors: Up, Right, Down, Left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    dr, dc = directions[action]
    new_head = (head[0] + dr, head[1] + dc)
    
    rows, cols = grid.shape
    row, col = new_head
    
    # Check bounds
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return new_head, True
    
    # Check if occupied
    if grid[row, col] != 0:
        return new_head, True
    
    return new_head, False
