"""Board ↔ pixel coordinate conversion and hit testing."""

import pygame
from src.config import BOARD_ORIGIN_X, BOARD_ORIGIN_Y, CELL_SIZE, HEIGHT, WIDTH


def cell_to_rect(row: int, col: int) -> pygame.Rect:
    """Return the pixel Rect for a given board cell."""
    x = BOARD_ORIGIN_X + col * CELL_SIZE
    y = BOARD_ORIGIN_Y + row * CELL_SIZE
    return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)


def pixel_to_cell(mouse_pos: tuple) -> tuple | None:
    """Convert a mouse position to (row, col), or None if outside the board."""
    mx, my = mouse_pos
    col = (mx - BOARD_ORIGIN_X) // CELL_SIZE
    row = (my - BOARD_ORIGIN_Y) // CELL_SIZE
    if 0 <= row < HEIGHT and 0 <= col < WIDTH:
        # Verify the click is actually inside the cell rect
        rect = cell_to_rect(row, col)
        if rect.collidepoint(mouse_pos):
            return (row, col)
    return None
