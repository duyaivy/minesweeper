"""Sidebar — status, AI Move button, auto-play checkbox, stats."""

import pygame
from src.config import (
    HEADER_HEIGHT,
    BOARD_AREA_W,
    SIDEBAR_W,
    BOARD_AREA_H,
    PANEL_BG,
    DARK_GRAY,
    LIGHT_GRAY,
    WHITE,
    RED_TEXT,
    CHECKBOX_CHECK,
)
from src.gui.widgets import draw_button, draw_checkbox


class Sidebar:
    """Draws the right-side control panel and exposes button rects."""

    def __init__(self, fonts: dict):
        self.medium = fonts["medium"]
        self.small = fonts["small"]
        self.tiny = fonts["tiny"]
        self.x = BOARD_AREA_W
        self.sx = self.x + 20

        self.ai_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.cb_rect = pygame.Rect(0, 0, 0, 0)

    def draw(self, surface: pygame.Surface, state, mouse_pos: tuple):
        # Background
        pygame.draw.rect(
            surface, PANEL_BG, (self.x, HEADER_HEIGHT, SIDEBAR_W, BOARD_AREA_H)
        )
        pygame.draw.line(
            surface,
            DARK_GRAY,
            (self.x, HEADER_HEIGHT),
            (self.x, HEADER_HEIGHT + BOARD_AREA_H),
            2,
        )

        y = HEADER_HEIGHT + 30

        # Status
        if state.lost:
            label, color = "Thua rồi!", RED_TEXT
        elif state.won:
            label, color = "Bạn thắng!", CHECKBOX_CHECK
        else:
            label, color = "Đang chơi...", WHITE
        surface.blit(self.medium.render(label, True, color), (self.sx, y))
        y += 60

        # AI Move button
        self.ai_btn_rect = pygame.Rect(self.sx, y, SIDEBAR_W - 40, 46)
        draw_button(surface, self.ai_btn_rect, "AI trợ giúp", self.small, mouse_pos)
        y += 70

        # AI auto-play checkbox
        self.cb_rect = draw_checkbox(
            surface,
            self.sx,
            y,
            state.ai_autoplay,
            "AI tự động chơi",
            self.small,
            mouse_pos,
        )
        y += 50

        # Stats
        ai = state.ai
        lines = [
            f"Đã mở: {len(state.revealed)}",
            f"Đánh dấu: {len(state.flags)}",
            f"AI biết an toàn: {len(ai.safes - ai.moves_made)}",
            f"AI biết mìn: {len(ai.mines)}",
        ]
        for line in lines:
            surface.blit(self.tiny.render(line, True, LIGHT_GRAY), (self.sx, y))
            y += 24
