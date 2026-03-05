"""Game rules and status evaluation — win/lose detection."""

import logging

log = logging.getLogger("minesweeper")


def evaluate_game_status(state, action_type: str = "unknown", cell=None) -> str:
    """Evaluate game end conditions and update state.
    
    Checks for win/lose conditions and updates state.won/state.lost flags.
    Should be called after every action (human reveal, flag, AI move).
    
    Args:
        state: GameState instance
        action_type: "reveal", "flag", "ai_step" for logging
        cell: (row, col) tuple for logging
        
    Returns:
        "playing" | "win" | "lose"
    """
    if state.is_game_over():
        return "win" if state.won else "lose"
    
    # Check win condition: all mines correctly flagged
    if not state.lost and state.game.mines == state.flags:
        state.won = True
        log.info("[GAME] status updated: playing -> win, cause=all_mines_flagged, "
                 "after action=%s, cell=%s", action_type, cell)
        return "win"
    
    # Check alternate win: all non-mine cells revealed
    total_cells = state.game.height * state.game.width
    total_mines = len(state.game.mines)
    if not state.lost and len(state.revealed) >= total_cells - total_mines:
        state.won = True
        log.info("[GAME] status updated: playing -> win, cause=all_safe_revealed, "
                 "after action=%s, cell=%s", action_type, cell)
        return "win"
    
    # Still playing
    return "playing"


def check_mine_hit(state, cell) -> bool:
    """Check if revealing a cell hits a mine, update state.lost if so.
    
    Args:
        state: GameState instance
        cell: (row, col) tuple
        
    Returns:
        True if mine hit, False otherwise
    """
    if state.game.is_mine(cell):
        state.lost = True
        log.info("[GAME] status updated: playing -> lose, cause=mine_hit, "
                 "cell=%s", cell)
        return True
    return False
