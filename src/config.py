"""All game constants: window, grid, colors, fonts, timing."""

import os

# Grid
HEIGHT = 15
WIDTH = 15
MINES = 40

# Window
WIN_W, WIN_H = 1028, 768
FPS = 30
HEADER_HEIGHT = 60
BOARD_PADDING = 20
SIDEBAR_RATIO = 0.28

# Derived layout
SIDEBAR_W = int(WIN_W * SIDEBAR_RATIO)
BOARD_AREA_W = WIN_W - SIDEBAR_W
BOARD_AREA_H = WIN_H - HEADER_HEIGHT

CELL_SIZE = int(
    min(
        (BOARD_AREA_W - BOARD_PADDING * 2) / WIDTH,
        (BOARD_AREA_H - BOARD_PADDING * 2) / HEIGHT,
    )
)

BOARD_PIXEL_W = CELL_SIZE * WIDTH
BOARD_PIXEL_H = CELL_SIZE * HEIGHT
BOARD_ORIGIN_X = (BOARD_AREA_W - BOARD_PIXEL_W) // 2
BOARD_ORIGIN_Y = HEADER_HEIGHT + (BOARD_AREA_H - BOARD_PIXEL_H) // 2

# AI timing
AI_TICK_MS = 300
AI_MAX_MOVES_PER_TURN = 1  # Số moves tối đa mỗi lượt autoplay (cân bằng logic vs UX)
CLICK_DELAY = 0.18

# Colors
BLACK = (0, 0, 0)
DARK_BG = (30, 30, 40)
GRAY = (180, 180, 180)
LIGHT_GRAY = (210, 210, 210)
DARK_GRAY = (120, 120, 120)
WHITE = (255, 255, 255)
HEADER_BG = (40, 44, 52)
PANEL_BG = (50, 54, 62)
BTN_COLOR = (70, 130, 230)
BTN_HOVER = (90, 150, 250)
BTN_TEXT = WHITE
RED_TEXT = (220, 50, 50)
TIMER_COLOR = (255, 220, 100)
MINE_COUNTER_COLOR = (255, 100, 100)

# Tile colors
CELL_UNREVEALED = (170, 175, 185)
CELL_UNREVEALED_BRD = (155, 160, 170)  # subtle border for unrevealed
CELL_REVEALED = (220, 225, 230)
CELL_REVEALED_BRD = (100, 105, 115)  # strong inset border for revealed
CELL_REVEALED_INNER = (190, 195, 200)  # inner shadow for inset look
CELL_EXPLODED = (255, 100, 100)  # red tint for exploded mine

# Hint / AI highlight
HINT_SAFE_COLOR = (0, 200, 80)
HINT_MINE_COLOR = (255, 160, 0)

# Checkbox
CHECKBOX_BG = (60, 64, 72)
CHECKBOX_CHECK = (0, 200, 80)

# Classic number colors
NUMBER_COLORS = {
    1: (0, 0, 255),
    2: (0, 128, 0),
    3: (255, 0, 0),
    4: (0, 0, 128),
    5: (128, 0, 0),
    6: (0, 128, 128),
    7: (0, 0, 0),
    8: (128, 128, 128),
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FONT_PATH = os.path.join(ASSETS_DIR, "fonts", "OpenSans-Regular.ttf")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")

# Logging
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    return logging.getLogger("minesweeper")
