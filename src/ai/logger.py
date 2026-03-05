"""Structured AI move logger — prints to console and stores entries."""

import time


class AILogger:
    """Logs every AI decision in a structured, readable format."""

    def __init__(self):
        self.entries = []
        self._index = 0

    def log(
        self, mode: str, action: str, cell: tuple | None, reason: str, game_status: str
    ):
        """Record one AI action.

        Args:
            mode: "autoplay" or "hint"
            action: "reveal" | "flag" | "guess" | "none"
            cell: (row, col) or None
            reason: short explanation
            game_status: "playing" | "win" | "lose"
        """
        self._index += 1
        entry = {
            "move_index": self._index,
            "mode": mode,
            "action": action,
            "cell": cell,
            "reason": reason,
            "timestamp": time.strftime("%H:%M:%S"),
            "game_status": game_status,
        }
        self.entries.append(entry)

        # Simple log without timestamp or details
        cell_str = f"({cell[0]},{cell[1]})" if cell else "none"
        if action in ["reveal", "guess", "flag"]:
            print(f"[{self._index}] {action} {cell_str}")

    def reset(self):
        self.entries.clear()
        self._index = 0

    def recent(self, n: int = 5) -> list[dict]:
        """Return the last *n* log entries (for optional GUI display)."""
        return self.entries[-n:]
