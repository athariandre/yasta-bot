# Code Review Fixes - Summary

This document summarizes the critical fixes applied based on the code review feedback.

## Commit: 767d146

### env.py Changes

#### 1. Added `who` Parameter for Switchable Observation Perspective
**Issue**: The environment always returned observations from p1's perspective, which breaks self-play in Workflow B.

**Fix**: 
- Added `who="p1"` parameter to `reset()` method
- Added `who="p1"` parameter to `step()` method
- Both methods now return observations from the specified player's perspective

**Verification**:
```python
env.reset(who='p1')  # Returns p1 perspective
env.reset(who='p2')  # Returns p2 perspective
env.step(action1, action2, who='p1')  # Returns p1 perspective
env.step(action1, action2, who='p2')  # Returns p2 perspective
```

#### 2. Fixed `legal_moves()` Edge Case
**Issue**: The function didn't handle the edge case where a player might compute a move that keeps them in the same position.

**Fix**: Added special case check:
```python
if new_pos == pos:
    legal.append(action)
    continue
```

#### 3. Added Clarifying Comment
**Issue**: Code reviewers might think heads being in both trail and head channels is a bug.

**Fix**: Added comment:
```python
# Heads remain part of the trail (occupied on grid), but we add separate
# head-only channels (3 and 4) for RL policy clarity. This is intentional.
```

---

### features.py Changes

#### 1. Fixed `reachable_area()` to Treat Head Cell as Empty
**Issue**: When evaluating moves, if the head position is marked on the grid, `reachable_area()` would return 0, causing degenerate evaluations in the heuristic bot.

**Fix**: Create a temporary grid where the head cell is treated as empty:
```python
temp_grid = grid.copy()
if temp_grid[head] != 0:
    temp_grid[head] = 0
```

**Impact**: The heuristic bot can now properly evaluate moves from any position, even if that position is already marked on the grid.

---

### heuristic_bot.py Changes

#### 1. Rewrote `_guess_opponent_move()` to Use Proper Environment Cloning
**Issue**: The function manually manipulated the grid without proper environment stepping, leading to inconsistent state transitions.

**Old Code**:
```python
temp_grid = env.grid.copy()
temp_grid[new_pos] = opp_id
area = reachable_area(temp_grid, new_pos)
```

**New Code**:
```python
cloned = env.clone()
if opp == 'p1':
    cloned.step(action, 0)
else:
    cloned.step(0, action)
opp_pos = cloned.p1_pos if opp == 'p1' else cloned.p2_pos
area = reachable_area(cloned.grid, opp_pos)
```

**Benefits**:
- Uses legal environment transitions
- Properly handles crashes
- Maintains trail consistency
- Follows simultaneous movement rules

#### 2. Fixed Cut Cells Calculation
**Issue**: Opponent area "before" was calculated inside the loop using potentially modified grid state.

**Fix**: Moved calculation to before the loop:
```python
# Compute opponent's reachable area before any move (for cut_cells calculation)
opp_pos_before = env.p1_pos if opp == 'p1' else env.p2_pos
area_opp_before = reachable_area(env.grid, opp_pos_before)
```

**Impact**: Ensures consistent baseline for measuring territorial cuts.

#### 3. Added Crash Detection in Opponent Move Guessing
**Issue**: Dead opponent moves weren't properly handled.

**Fix**: Added check:
```python
opp_dead = cloned.p1_dead if opp == 'p1' else cloned.p2_dead
if opp_dead:
    area = 0
else:
    area = reachable_area(cloned.grid, opp_pos)
```

---

## Verification

All changes have been verified:
- ✅ All 15 original tests pass
- ✅ New functionality tested (switchable perspectives)
- ✅ Determinism maintained
- ✅ No security vulnerabilities (CodeQL: 0 alerts)
- ✅ Backward compatible (default `who='p1'` parameter)

## Impact on Workflow B

These fixes are **critical** for Workflow B (RL training):
1. Self-play requires both p1 and p2 perspectives ✅
2. Heuristic bot must provide high-quality behavior cloning data ✅
3. Environment must handle all edge cases correctly ✅
4. All components must maintain determinism ✅

The environment is now ready for reinforcement learning training.
