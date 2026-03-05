"""MinesweeperAI — constraint-based reasoning agent + ML fallback."""

import random
import logging
from src.game.sentence import Sentence

log = logging.getLogger("minesweeper")


class MinesweeperAI:
    """
    AI player that tracks knowledge as a list of Sentences.
    When rule-based logic has no safe move, delegates to MLPredictor
    instead of picking randomly.
    """

    def __init__(self, height=8, width=8, ml_predictor=None):
        self.height = height
        self.width = width
        self.moves_made = set()
        self.mines = set()
        self.safes = set()
        self.knowledge = []
        self.ml_predictor = ml_predictor

    def mark_mine(self, cell):
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def nearby_cells(self, cell):
        cells = set()
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if (i, j) == cell:
                    continue
                if 0 <= i < self.height and 0 <= j < self.width:
                    cells.add((i, j))
        return cells

    def add_knowledge(self, cell, count):
        """Record that *cell* has *count* neighboring mines, then infer."""
        self.moves_made.add(cell)

        if cell not in self.safes:
            self.mark_safe(cell)

        nearby = self.nearby_cells(cell)
        nearby -= self.safes | self.moves_made
        new_sentence = Sentence(nearby, count)
        self.knowledge.append(new_sentence)

        new_safes = set()
        new_mines = set()
        for sentence in self.knowledge:
            if len(sentence.cells) == 0:
                self.knowledge.remove(sentence)
            else:
                tmp_safes = sentence.known_safes()
                tmp_mines = sentence.known_mines()
                if isinstance(tmp_safes, set):
                    new_safes |= tmp_safes
                if isinstance(tmp_mines, set):
                    new_mines |= tmp_mines

        for safe in new_safes:
            self.mark_safe(safe)
        for mine in new_mines:
            self.mark_mine(mine)

        prev = new_sentence
        new_inferences = []
        for sentence in self.knowledge:
            if len(sentence.cells) == 0:
                self.knowledge.remove(sentence)
            elif prev == sentence:
                break
            elif prev.cells <= sentence.cells:
                inf_cells = sentence.cells - prev.cells
                inf_count = sentence.count - prev.count
                new_inferences.append(Sentence(inf_cells, inf_count))
            prev = sentence
        self.knowledge += new_inferences

    def make_safe_move(self):
        """Return a known-safe cell not yet revealed, or None."""
        safe_moves = self.safes - self.moves_made
        return safe_moves.pop() if safe_moves else None

    def make_random_move(self):
        """
        Return a move for the 'must guess' situation.
        - If MLPredictor available → dùng ML.
        - Fallback → random.

        Fix 1: revealed truyền vào ML = moves_made ONLY
                (không gộp safes — safes là ô AI biết an toàn nhưng chưa lật,
                 ML sẽ bị nhầm tưởng ô đó đã có số hiển thị)

        Fix 2: ưu tiên border candidates (ô kề vùng đã lật)
                → ML có đủ context số xung quanh để dự đoán chính xác
                → tránh tình huống 63 candidates mà phần lớn cô lập
        """
        total = self.height * self.width
        if len(self.moves_made) >= total - len(self.mines):
            return None

        not_allowed = self.moves_made | self.mines
        all_candidates = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in not_allowed
        ]
        if not all_candidates:
            return None

        # Fix 2: border = ô chưa lật kề ít nhất 1 ô đã thực sự lật
        border_candidates = [
            cell
            for cell in all_candidates
            if any(nb in self.moves_made for nb in self.nearby_cells(cell))
        ]
        candidates = border_candidates if border_candidates else all_candidates

        if self.ml_predictor is not None and self.ml_predictor.trained:
            # Use ML predictor for best guess
            cell, prob = self.ml_predictor.predict_safest(
                self._board_ref, self.moves_made, candidates, self.mines
            )
            # Store prob for logging
            self._last_ml_prob = prob
            return cell

        # Fallback: chọn random nếu không có AI
        self._last_ml_prob = None
        if not candidates:
            return None
        return random.choice(candidates)

    @property
    def _board_ref(self):
        return getattr(self, "_board", None)

    @_board_ref.setter
    def _board_ref(self, board):
        self._board = board
