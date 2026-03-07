"""Minesweeper board data model — mine placement and neighbor counting."""

import random
from collections import deque


class Minesweeper:
    """Minesweeper game representation."""

    def __init__(self, height=8, width=8, mines=8):
        self.height = height
        self.width = width
        self.mines = set()
        self.mines_found = set()

        self.board = [[False] * self.width for _ in range(self.height)]

        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

    def print(self):
        """Text-based representation for debugging."""
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                print("|X" if self.board[i][j] else "| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """Count mines in the 8 neighbors of *cell*."""
        count = 0
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if (i, j) == cell:
                    continue
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1
        return count

    def won(self):
        return self.mines_found == self.mines

    def reveal_flood_fill(self, start_cell, revealed: set, flags: set):
        if self.nearby_mines(start_cell) != 0:
            if start_cell not in revealed and start_cell not in flags:
                revealed.add(start_cell)
                return {start_cell}
            return set()
        newly_opened = set()
        queue = deque([start_cell])
        visited = {start_cell}
        while queue:
            cell = queue.popleft()
            if cell in revealed or cell in flags:
                continue
            # Mo o hien tai
            revealed.add(cell)
            newly_opened.add(cell)
            count = self.nearby_mines(cell)
            if count > 0:
                continue
            for i in range(cell[0] - 1, cell[0] + 2):
                for j in range(cell[1] - 1, cell[1] + 2):
                    neighbor = (i, j)
                    if neighbor == cell:
                        continue
                    if 0 <= i < self.height and 0 <= j < self.width:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

        return newly_opened
