"""
Deterministic heuristic bot for Tron.
Part of Workflow A - Core Engine Implementation.

This bot will be used for behavior cloning in RL training.
"""

from features import reachable_area, voronoi_counts, is_tunnel


def choose_action(env, me):
    """
    Choose the best action using deterministic heuristic evaluation.
    
    Args:
        env: TronEnv instance
        me: 'p1' or 'p2'
        
    Returns:
        Integer action (0=Up, 1=Right, 2=Down, 3=Left)
    """
    legal = env.legal_moves(me)
    
    # If no legal moves, return 0 (will crash anyway)
    if not legal:
        return 0
    
    # If only one legal move, return it
    if len(legal) == 1:
        return legal[0]
    
    # Determine opponent
    opp = 'p2' if me == 'p1' else 'p1'
    
    # Compute opponent's reachable area before any move (for cut_cells calculation)
    opp_pos_before = env.p1_pos if opp == 'p1' else env.p2_pos
    area_opp_before = reachable_area(env.grid, opp_pos_before)
    
    # Evaluate each legal move
    best_action = None
    best_score = None
    best_voronoi_diff = None
    best_next_mobility = None
    
    for action in legal:
        # Clone environment
        cloned_env = env.clone()
        
        # Simulate my move
        my_action = action
        
        # Guess opponent's move: action that maximizes their reachable area
        opp_legal = cloned_env.legal_moves(opp)
        opp_action = _guess_opponent_move(cloned_env, opp, opp_legal)
        
        # Step cloned environment
        if me == 'p1':
            _, _, _, done, _ = cloned_env.step(my_action, opp_action)
        else:
            _, _, _, done, _ = cloned_env.step(opp_action, my_action)
        
        # If this move leads to immediate death, score it very low
        my_dead = cloned_env.p1_dead if me == 'p1' else cloned_env.p2_dead
        if my_dead:
            score = -1000000
            voronoi_diff = -1000000
            next_mobility = 0
        else:
            # Get positions after move
            my_pos = cloned_env.p1_pos if me == 'p1' else cloned_env.p2_pos
            opp_pos = cloned_env.p1_pos if opp == 'p1' else cloned_env.p2_pos
            
            # Compute areas
            area_me = reachable_area(cloned_env.grid, my_pos)
            area_opp = reachable_area(cloned_env.grid, opp_pos)
            
            # Tunnel penalty: check if my new head is in a tunnel
            tunnel_penalty = 0
            if is_tunnel(cloned_env.grid, my_pos):
                tunnel_penalty = 5
            
            # Cut bonus: cells reachable by opponent before but not after
            cut_cells = max(0, area_opp_before - area_opp)
            cut_bonus = 1 * cut_cells
            
            # Compute score
            score = area_me - area_opp - tunnel_penalty + cut_bonus
            
            # Compute Voronoi for tie-breaking
            voronoi_me, voronoi_opp, _ = voronoi_counts(cloned_env.grid, my_pos, opp_pos)
            voronoi_diff = voronoi_me - voronoi_opp
            
            # Compute next-turn mobility for tie-breaking
            next_mobility = len(cloned_env.legal_moves(me))
        
        # Update best action with tie-breaking
        if best_action is None:
            best_action = action
            best_score = score
            best_voronoi_diff = voronoi_diff
            best_next_mobility = next_mobility
        else:
            # Compare scores
            if score > best_score:
                best_action = action
                best_score = score
                best_voronoi_diff = voronoi_diff
                best_next_mobility = next_mobility
            elif score == best_score:
                # Tie-break by Voronoi advantage
                if voronoi_diff > best_voronoi_diff:
                    best_action = action
                    best_score = score
                    best_voronoi_diff = voronoi_diff
                    best_next_mobility = next_mobility
                elif voronoi_diff == best_voronoi_diff:
                    # Tie-break by next-turn mobility
                    if next_mobility > best_next_mobility:
                        best_action = action
                        best_score = score
                        best_voronoi_diff = voronoi_diff
                        best_next_mobility = next_mobility
                    elif next_mobility == best_next_mobility:
                        # Deterministic fallback: prefer lower action ID
                        if action < best_action:
                            best_action = action
                            best_score = score
                            best_voronoi_diff = voronoi_diff
                            best_next_mobility = next_mobility
    
    return best_action


def _guess_opponent_move(env, opp, opp_legal):
    """
    Guess opponent's move: action that maximizes their reachable area.
    
    Uses proper environment cloning and stepping to simulate opponent moves.
    
    Args:
        env: TronEnv instance
        opp: 'p1' or 'p2'
        opp_legal: List of legal actions for opponent
        
    Returns:
        Integer action
    """
    if not opp_legal:
        return 0
    
    if len(opp_legal) == 1:
        return opp_legal[0]
    
    best_action = None
    best_area = None
    
    for action in opp_legal:
        # Clone environment to simulate opponent's move
        cloned = env.clone()
        
        # Step environment with opponent action
        # Use dummy action (0) for the other player
        if opp == 'p1':
            # Opponent is p1, so action_my = action, action_opp = 0 (dummy)
            cloned.step(action, 0)
        else:
            # Opponent is p2, so action_my = 0 (dummy), action_opp = action
            cloned.step(0, action)
        
        # Get opponent's position after move
        opp_pos = cloned.p1_pos if opp == 'p1' else cloned.p2_pos
        
        # Check if opponent crashed
        opp_dead = cloned.p1_dead if opp == 'p1' else cloned.p2_dead
        if opp_dead:
            # Dead move has area 0
            area = 0
        else:
            # Compute reachable area from new position
            area = reachable_area(cloned.grid, opp_pos)
        
        # Update best action
        if best_action is None or area > best_area:
            best_action = action
            best_area = area
        elif area == best_area:
            # Tie-break by lower action ID for determinism
            if action < best_action:
                best_action = action
                best_area = area
    
    return best_action
