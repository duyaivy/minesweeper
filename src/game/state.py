"""Game state container — single source of truth for a running game."""

import time as _time
import logging
from src.config import HEIGHT, WIDTH, MINES
from src.game.board import Minesweeper
from src.ai.agent import MinesweeperAI
from src.ai.logger import AILogger

log = logging.getLogger("minesweeper")


class GameState:
    """Holds all mutable state for one game session."""

    # Class-level ML predictor — shared across resets, trained once at boot
    _ml_predictor = None

    def __init__(self):
        self.game = Minesweeper(height=HEIGHT, width=WIDTH, mines=MINES)
        self.ai = MinesweeperAI(
            height=HEIGHT,
            width=WIDTH,
            ml_predictor=GameState._ml_predictor,
        )
        # Give the AI a reference to the board so ML can query it
        self.ai._board = self.game

        self.ai_logger = AILogger()
        self.revealed: set = set()
        self.flags: set = set()
        self.lost: bool = False
        self.won: bool = False
        self.ai_autoplay: bool = False
        self.start_time: float | None = None
        self.elapsed: int = 0

    # ── ML helpers ──────────────────────────────────────

    @classmethod
    def set_ml_predictor(cls, predictor):
        """Called once from main() after ML training completes."""
        cls._ml_predictor = predictor
        if predictor is not None:
            log.info(
                "[STATE] MLPredictor attached (val_acc=%.4f)", predictor.val_accuracy
            )
        else:
            log.info("[STATE] MLPredictor disabled - using random fallback")

    @property
    def ml_ready(self) -> bool:
        return GameState._ml_predictor is not None and GameState._ml_predictor.trained

    @property
    def ml_accuracy(self) -> float:
        if self.ml_ready:
            return GameState._ml_predictor.val_accuracy
        return 0.0

    # ── Helpers ─────────────────────────────────────────

    def is_game_over(self) -> bool:
        return self.lost or self.won

    def start_timer_if_needed(self):
        if self.start_time is None:
            self.start_time = _time.time()

    def tick_timer(self):
        if self.start_time is not None and not self.is_game_over():
            self.elapsed = int(_time.time() - self.start_time)

    def check_win(self):
        if not self.lost and self.game.mines == self.flags:
            self.won = True
