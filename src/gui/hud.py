"""Header bar — timer (with clock icon), mine counter, reset button."""

import pygame
from src.config import (
    WIN_W, HEADER_HEIGHT, BOARD_AREA_W,
    HEADER_BG, MINE_COUNTER_COLOR, TIMER_COLOR,
)
from src.gui.widgets import draw_button
from src.gui.assets import assets


class HUD:
    """Draws the top header bar and exposes button rects for hit testing."""

    def __init__(self, fonts: dict):
        self.medium = fonts["medium"]
        self.small = fonts["small"]
        self.reset_rect = pygame.Rect(WIN_W - 150, 10, 130, HEADER_HEIGHT - 20)

        # Try to load clock icon (32×32); fall back to text if missing
        self._clock_icon = assets.get_icon("clock.png", 30)

    def draw(self, surface: pygame.Surface, mines_remaining: int,
             elapsed: int, mouse_pos: tuple):
        pygame.draw.rect(surface, HEADER_BG, (0, 0, WIN_W, HEADER_HEIGHT))

        # Mine counter (left)
        mc = self.medium.render(f"  {mines_remaining}", True, MINE_COUNTER_COLOR)
        mine_icon = assets.get_icon("mine.png", 26)
        y_center = (HEADER_HEIGHT - mc.get_height()) // 2
        if mine_icon:
            surface.blit(mine_icon, (16, (HEADER_HEIGHT - 26) // 2))
            surface.blit(mc, (44, y_center))
        else:
            mc_full = self.medium.render(f"Mìn: {mines_remaining}", True, MINE_COUNTER_COLOR)
            surface.blit(mc_full, (16, y_center))

        # Timer (center of board area, with clock icon)
        time_str = f" {elapsed}s"
        timer_surf = self.medium.render(time_str, True, TIMER_COLOR)
        tx = BOARD_AREA_W // 2 - (timer_surf.get_width() + 34) // 2
        ty = (HEADER_HEIGHT - timer_surf.get_height()) // 2
        if self._clock_icon:
            surface.blit(self._clock_icon, (tx, (HEADER_HEIGHT - 30) // 2))
            surface.blit(timer_surf, (tx + 32, ty))
        else:
            fallback = self.medium.render(f"Thời gian: {elapsed}s", True, TIMER_COLOR)
            surface.blit(fallback, (tx, ty))

        # Reset button (right)
        draw_button(surface, self.reset_rect, "Chơi Lại", self.small, mouse_pos)
