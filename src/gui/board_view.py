"""Board view — draws the tile grid, numbers, flags, mines, and hint overlays."""

import pygame
from src.config import (
    HEIGHT, WIDTH, CELL_SIZE,
    BLACK, DARK_GRAY,
    CELL_UNREVEALED, CELL_UNREVEALED_BRD,
    CELL_REVEALED, CELL_REVEALED_BRD, CELL_REVEALED_INNER,
    CELL_EXPLODED,
    HINT_SAFE_COLOR, HINT_MINE_COLOR,
    NUMBER_COLORS,
)
from src.utils.geometry import cell_to_rect
from src.gui.assets import assets


class BoardView:
    """Renders the Minesweeper grid."""

    def __init__(self, fonts: dict):
        self.cell_font = fonts["cell"]
        self._flag_img = assets.get_cell_image("flag.png")
        self._mine_img = assets.get_cell_image("mine.png")
        # Pre-compute cell rects (static grid)
        self.rects = [
            [cell_to_rect(i, j) for j in range(WIDTH)]
            for i in range(HEIGHT)
        ]

    def draw(self, surface: pygame.Surface, state):
        """Draw all tiles, content, and hint overlays."""
        for i in range(HEIGHT):
            for j in range(WIDTH):
                rect = self.rects[i][j]
                cell = (i, j)
                revealed = cell in state.revealed
                flagged = cell in state.flags
                is_mine = state.game.is_mine(cell)

                # --- Tile background & border ---
                if revealed:
                    pygame.draw.rect(surface, CELL_REVEALED, rect)
                    # Strong inset border: dark edges + inner shadow on top-left
                    pygame.draw.rect(surface, CELL_REVEALED_BRD, rect, 2)
                    # Inner shadow lines (top & left, 1px inset)
                    inner = rect.inflate(-4, -4)
                    pygame.draw.line(surface, CELL_REVEALED_INNER,
                                     inner.topleft, inner.topright, 1)
                    pygame.draw.line(surface, CELL_REVEALED_INNER,
                                     inner.topleft, inner.bottomleft, 1)
                elif is_mine and state.lost:
                    # Exploded: red tint
                    pygame.draw.rect(surface, CELL_EXPLODED, rect)
                    pygame.draw.rect(surface, CELL_REVEALED_BRD, rect, 2)
                else:
                    # Unrevealed / flagged: subtle light border
                    pygame.draw.rect(surface, CELL_UNREVEALED, rect)
                    pygame.draw.rect(surface, CELL_UNREVEALED_BRD, rect, 1)

                # --- Content ---
                if is_mine and state.lost:
                    if self._mine_img:
                        surface.blit(self._mine_img, rect)
                elif flagged:
                    if self._flag_img:
                        surface.blit(self._flag_img, rect)
                elif revealed:
                    count = state.game.nearby_mines(cell)
                    if count > 0:
                        color = NUMBER_COLORS.get(count, BLACK)
                        num = self.cell_font.render(str(count), True, color)
                        surface.blit(num, num.get_rect(center=rect.center))

    def draw_hints(self, surface: pygame.Surface, state):
        """Draw hint overlays when AI auto-play is OFF."""
        if state.ai_autoplay or state.is_game_over():
            return

        for cell in (state.ai.safes - state.ai.moves_made):
            pygame.draw.rect(surface, HINT_SAFE_COLOR, self.rects[cell[0]][cell[1]], 3)

        for cell in (state.ai.mines - state.flags):
            pygame.draw.rect(surface, HINT_MINE_COLOR, self.rects[cell[0]][cell[1]], 3)
