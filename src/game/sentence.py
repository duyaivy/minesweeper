"""Logical sentence for constraint-based Minesweeper reasoning."""


class Sentence:
    """A set of board cells and a count of how many are mines."""

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """If cell count equals mine count, all cells are mines."""
        if len(self.cells) == self.count:
            return self.cells

    def known_safes(self):
        """If mine count is 0, all cells are safe."""
        if self.count == 0:
            return self.cells

    def mark_mine(self, cell):
        if cell in self.cells:
            self.cells.discard(cell)
            self.count -= 1

    def mark_safe(self, cell):
        if cell in self.cells:
            self.cells.discard(cell)
