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
        """
        Record that *cell* has *count* neighboring mines, then infer.

        Uses iterative inference: loop until no new mines/safes are discovered.
        This maximizes logical deduction and minimizes random guesses.
        """
        self.moves_made.add(cell)

        if cell not in self.safes:
            self.mark_safe(cell)

        # Tạo sentence mới từ các ô láng giềng chưa biết
        nearby = self.nearby_cells(cell)
        nearby -= self.safes | self.moves_made
        new_sentence = Sentence(nearby, count)

        # Loại bỏ known mines khỏi sentence mới
        for mine in self.mines:
            if mine in new_sentence.cells:
                new_sentence.cells.discard(mine)
                new_sentence.count -= 1

        # Only add non-empty sentences
        if len(new_sentence.cells) > 0 or new_sentence.count > 0:
            self.knowledge.append(new_sentence)

        # ITERATIVE INFERENCE: lặp cho đến khi không còn suy luận mới
        max_iterations = 100  # Safety limit
        iteration = 0
        changed = True

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            # 1) Cleanup: Xóa sentences rỗng
            self.knowledge = [s for s in self.knowledge if len(s.cells) > 0]

            # 2) Direct inference: Check known_mines và known_safes
            new_safes = set()
            new_mines = set()

            for sentence in self.knowledge:
                tmp_safes = sentence.known_safes()
                tmp_mines = sentence.known_mines()

                if tmp_safes:
                    new_safes |= tmp_safes
                if tmp_mines:
                    new_mines |= tmp_mines

            # Mark tất cả safes mới phát hiện
            for safe in new_safes:
                if safe not in self.safes:
                    self.mark_safe(safe)
                    changed = True

            # Mark tất cả mines mới phát hiện
            for mine in new_mines:
                if mine not in self.mines:
                    self.mark_mine(mine)
                    changed = True

            # 3) Subset inference: Kiểm tra tất cả cặp sentences
            new_inferences = []
            for s1 in self.knowledge:
                for s2 in self.knowledge:
                    if s1 == s2:
                        continue

                    # Nếu s1 là tập con của s2
                    if s1.cells and s2.cells and s1.cells <= s2.cells:
                        # s2 - s1 = difference
                        inf_cells = s2.cells - s1.cells
                        inf_count = s2.count - s1.count

                        if inf_cells:  # Only create non-empty inferences
                            inference = Sentence(inf_cells, inf_count)
                            # Kiểm tra xem inference đã tồn tại chưa
                            if (
                                inference not in self.knowledge
                                and inference not in new_inferences
                            ):
                                new_inferences.append(inference)
                                changed = True

            # Thêm các inferences mới vào knowledge base
            self.knowledge.extend(new_inferences)

    def make_safe_move(self):
        """Return a known-safe cell not yet revealed, or None."""
        safe_moves = self.safes - self.moves_made
        return safe_moves.pop() if safe_moves else None

    def make_random_move(self):
        """
        Return a move for the 'must guess' situation.
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

        border_candidates = [
            cell
            for cell in all_candidates
            if any(nb in self.moves_made for nb in self.nearby_cells(cell))
        ]
        candidates = border_candidates if border_candidates else all_candidates

        if self.ml_predictor is not None and self.ml_predictor.trained:
            cell, prob = self.ml_predictor.predict_safest(
                self._board_ref, self.moves_made, candidates, self.mines
            )
            self._last_ml_prob = prob
            return cell

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
