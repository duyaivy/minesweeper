"""Reusable UI primitives — button and checkbox drawing."""

import pygame
from src.config import (
    BTN_COLOR, BTN_HOVER, BTN_TEXT, WHITE,
    CHECKBOX_BG, CHECKBOX_CHECK,
)


def draw_button(
    surface: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    font: pygame.font.Font,
    mouse_pos: tuple,
    bg=BTN_COLOR,
    hover=BTN_HOVER,
    fg=BTN_TEXT,
) -> pygame.Rect:
    """Draw a rounded button; return its rect for hit testing."""
    color = hover if rect.collidepoint(mouse_pos) else bg
    pygame.draw.rect(surface, color, rect, border_radius=6)
    txt = font.render(text, True, fg)
    surface.blit(txt, txt.get_rect(center=rect.center))
    return rect


def draw_checkbox(
    surface: pygame.Surface,
    x: int, y: int,
    checked: bool,
    label: str,
    font: pygame.font.Font,
    mouse_pos: tuple,
) -> pygame.Rect:
    """Draw a checkbox with label; return the full clickable rect."""
    box = 22
    box_rect = pygame.Rect(x, y, box, box)
    pygame.draw.rect(surface, CHECKBOX_BG, box_rect, border_radius=3)
    pygame.draw.rect(surface, WHITE, box_rect, 2, border_radius=3)
    if checked:
        inner = box_rect.inflate(-6, -6)
        pygame.draw.rect(surface, CHECKBOX_CHECK, inner, border_radius=2)
    lbl = font.render(label, True, WHITE)
    surface.blit(lbl, (x + box + 8, y + (box - lbl.get_height()) // 2))
    return pygame.Rect(x, y, box + 8 + lbl.get_width(), box)
