"""XGBoost ML predictor — trains on simulated games, predicts mine probability."""

import random
import logging
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

log = logging.getLogger("minesweeper")


# ─────────────────────────────────────────────────────────
# Feature extraction  (14 features per unrevealed cell)
# ─────────────────────────────────────────────────────────


def extract_features(
    board,
    height: int,
    width: int,
    row: int,
    col: int,
    revealed: set,
    mines_known: set = None,
) -> list:
    """
    Build a 14-element feature vector for cell (row, col).

    Parameters:
      - revealed: set of cells that have been revealed (clicked)
      - mines_known: set of cells flagged/known as mines by AI

    Features:
      0  bias
      1  sum_mine_probs       – weighted probability from number neighbours
      2  mines_around         – confirmed mines already flagged nearby
      3  revealed_around      – count of revealed neighbours
      4  unrevealed_around    – count of still-hidden neighbours
      5  max_number_around    – highest clue value adjacent
      6  min_number_around    – lowest clue value adjacent (0 if none)
      7  avg_number_around    – mean clue value adjacent
      8  ratio_mines          – mines_around / revealed_around
      9  is_corner
      10 is_edge
      11 zero_cells_around    – revealed cells with value 0 nearby
      12 dist_center          – normalised Manhattan distance to board centre
      13 sum_remaining_mines  – Σ (clue − known_mines) for number neighbours
    """
    if mines_known is None:
        mines_known = set()
    row_range = range(max(row - 1, 0), min(row + 1, height - 1) + 1)
    col_range = range(max(col - 1, 0), min(col + 1, width - 1) + 1)

    mines_around = 0
    revealed_around = 0
    unrevealed_around = 0
    zero_cells = 0
    numbers = []
    sum_probs = 0.0
    sum_remaining = 0.0

    for r in row_range:
        for c in col_range:
            if (r, c) == (row, col):
                continue
            if (r, c) in revealed:
                revealed_around += 1
                val = board.nearby_mines((r, c))  # clue number
                if val == 0:
                    zero_cells += 1
                else:
                    numbers.append(val)
                    # probability contribution
                    stats = _neighbour_stats(
                        board, height, width, r, c, revealed, mines_known
                    )
                    if stats["unrevealed"] > 0:
                        remaining = val - stats["mines"]
                        p = remaining / stats["unrevealed"]
                        sum_probs += max(0.0, p)
                        sum_remaining += max(0.0, remaining)
            elif (r, c) in mines_known:
                # Count flagged mines
                mines_around += 1
            else:
                unrevealed_around += 1

    max_num = max(numbers) if numbers else 0
    min_num = min(numbers) if numbers else 0
    avg_num = sum(numbers) / len(numbers) if numbers else 0.0

    ratio = mines_around / revealed_around if revealed_around > 0 else 0.0

    is_corner = int((row in (0, height - 1)) and (col in (0, width - 1)))
    is_edge = int((row in (0, height - 1)) or (col in (0, width - 1))) - is_corner

    cx, cy = height / 2.0, width / 2.0
    max_d = (height + width) / 2.0
    dist_c = (abs(row - cx) + abs(col - cy)) / max_d

    return [
        1,  # 0 bias
        sum_probs,  # 1
        mines_around,  # 2
        revealed_around,  # 3
        unrevealed_around,  # 4
        max_num,  # 5
        min_num,  # 6
        avg_num,  # 7
        ratio,  # 8
        is_corner,  # 9
        is_edge,  # 10
        zero_cells,  # 11
        dist_c,  # 12
        sum_remaining,  # 13
    ]


def _neighbour_stats(board, height, width, row, col, revealed, mines_known):
    """Count flagged mines & unrevealed cells around (row, col)."""
    mines = unrevealed = 0
    for r in range(max(row - 1, 0), min(row + 1, height - 1) + 1):
        for c in range(max(col - 1, 0), min(col + 1, width - 1) + 1):
            if (r, c) == (row, col):
                continue
            if (r, c) in mines_known:
                # Count flagged/known mines
                mines += 1
            elif (r, c) not in revealed:
                # Count unrevealed (not yet clicked, not flagged)
                unrevealed += 1
    return {"mines": mines, "unrevealed": unrevealed}


# ─────────────────────────────────────────────────────────
# Simulated-game data collector
# ─────────────────────────────────────────────────────────


def _collect_one_game(height, width, mines_count):
    """
    Simulate one game with random moves and collect (features, label) pairs
    at every point where we must guess (no safe move available).

    FIX: Only collect border cells (cells adjacent to revealed cells)
         to match real game behavior where AI prioritizes border candidates.
    """
    from src.game.board import Minesweeper

    board = Minesweeper(height=height, width=width, mines=mines_count)

    all_cells = [(r, c) for r in range(height) for c in range(width)]
    revealed = set()
    safe_known = set()
    mines_known = set()

    # ── simple rule-based helper ──────────────────────────
    def rule_step():
        """Return a guaranteed-safe cell if one can be deduced, else None."""
        for r, c in list(revealed):
            clue = board.nearby_mines((r, c))
            nbrs = [
                (r2, c2)
                for r2 in range(max(r - 1, 0), min(r + 1, height - 1) + 1)
                for c2 in range(max(c - 1, 0), min(c + 1, width - 1) + 1)
                if (r2, c2) != (r, c)
            ]
            hidden = [n for n in nbrs if n not in revealed]
            flagged = [n for n in nbrs if n in mines_known]
            if len(flagged) == clue:  # all mines accounted for
                for n in hidden:
                    if n not in mines_known:
                        return n  # safe!
            if len(hidden) == clue - len(flagged):  # all hidden must be mines
                for n in hidden:
                    mines_known.add(n)
        return None

    def get_neighbors(cell):
        """Get all valid neighbor cells."""
        r, c = cell
        neighbors = []
        for i in range(max(r - 1, 0), min(r + 1, height - 1) + 1):
            for j in range(max(c - 1, 0), min(c + 1, width - 1) + 1):
                if (i, j) != (r, c):
                    neighbors.append((i, j))
        return neighbors

    # First click: pick a non-mine cell
    non_mines = [c for c in all_cells if not board.is_mine(c)]
    start = random.choice(non_mines)
    revealed.add(start)

    xs, ys = [], []

    for _ in range(height * width):
        if len(revealed) + len(mines_known) >= len(all_cells):
            break

        safe = rule_step()
        if safe and safe not in revealed:
            revealed.add(safe)
            continue

        # ── must guess: collect training data ──
        all_candidates = [
            c for c in all_cells if c not in revealed and c not in mines_known
        ]
        if not all_candidates:
            break

        # FIX: Only use border cells (cells next to revealed cells)
        # This matches real game behavior in agent.py
        border_candidates = [
            cell
            for cell in all_candidates
            if any(nb in revealed for nb in get_neighbors(cell))
        ]

        # Use border if available, otherwise fallback to all (rare in real games)
        candidates = border_candidates if border_candidates else all_candidates

        # Collect features for candidates (now much smaller set!)
        for cell in candidates:
            feat = extract_features(
                board, height, width, cell[0], cell[1], revealed, mines_known
            )
            label = 1 if board.is_mine(cell) else 0
            xs.append(feat)
            ys.append(label)

        # Random pick to continue simulation
        pick = random.choice(candidates)
        if board.is_mine(pick):
            break  # game over in simulation
        revealed.add(pick)

    return xs, ys


# ─────────────────────────────────────────────────────────
# ML Model
# ─────────────────────────────────────────────────────────


class MLPredictor:
    """
    XGBoost model predicting mine probability for unrevealed cells.

    Usage:
        predictor = MLPredictor(height=10, width=10, mines=15)
        predictor.train(n_games=1000)
        safest = predictor.predict_safest(board, revealed, candidates, mines_known)
    """

    FEATURE_NAMES = [
        "bias",
        "sum_probs",
        "mines_around",
        "revealed_around",
        "unrevealed_around",
        "max_number",
        "min_number",
        "avg_number",
        "ratio_mines",
        "is_corner",
        "is_edge",
        "zero_cells",
        "dist_center",
        "sum_remaining",
    ]

    def __init__(self, height: int = 10, width: int = 10, mines: int = 15):
        self.height = height
        self.width = width
        self.mines = mines
        self.trained = False
        self.val_accuracy = 0.0

        # Improved XGBoost parameters for better mine prediction
        # - scale_pos_weight: handle class imbalance (mines are minority class)
        # - increased n_estimators & depth for more complex patterns
        # - lower learning_rate for better generalization
        safe_to_mine_ratio = (height * width - mines) / mines

        self.clf = XGBClassifier(
            n_estimators=500,  # More trees for better learning
            max_depth=8,  # Deeper trees for complex patterns
            learning_rate=0.03,  # Lower LR for better generalization
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=safe_to_mine_ratio,  # Balance classes
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
            min_child_weight=1,  # Allow learning from small groups
            gamma=0.1,  # Min loss reduction for split
        )

    # ── Training ─────────────────────────────────────────

    def train(self, n_games: int = 1000) -> None:
        print(f"Collecting data from {n_games} games...")
        X_list, y_list = [], []

        for i in range(n_games):
            xs, ys = _collect_one_game(self.height, self.width, self.mines)
            X_list.extend(xs)
            y_list.extend(ys)
            if (i + 1) % 200 == 0:
                print(f"  Progress: {i+1}/{n_games}")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        n_mine = int(y.sum())
        n_safe = int((y == 0).sum())
        print(f"Total samples: {len(y)} (mines: {n_mine}, safe: {n_safe})")

        if len(y) < 50:
            n_samples = len(y)
            print(f"Warning: Low data ({n_samples} samples)")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        # Detailed evaluation metrics
        y_pred = self.clf.predict(X_val)
        y_proba = self.clf.predict_proba(X_val)[:, 1]

        from sklearn.metrics import precision_score, recall_score, roc_auc_score

        self.val_accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_val, y_proba)
        except:
            auc = 0.0

        self.trained = True

        print(f"Training complete! Accuracy: {self.val_accuracy*100:.1f}%")

    # ── Prediction ───────────────────────────────────────

    def predict_safest(
        self, board, revealed: set, candidates: list, mines_known: set = None
    ):
        """
        Return the candidate cell with the lowest predicted mine probability.
        Falls back to random if model not trained or no candidates.

        Parameters:
          - revealed: set of cells that have been revealed
          - candidates: list of cells to choose from
          - mines_known: set of cells flagged/known as mines
        """
        if not candidates:
            return None, None
        if not self.trained:
            # Model chưa train, chọn random
            return random.choice(candidates), None

        if mines_known is None:
            mines_known = set()

        feats = [
            extract_features(
                board, self.height, self.width, r, c, revealed, mines_known
            )
            for (r, c) in candidates
        ]
        X = np.array(feats, dtype=np.float32)
        probs = self.clf.predict_proba(X)[:, 1]  # P(mine)

        best_idx = int(np.argmin(probs))
        best_cell = candidates[best_idx]
        best_prob = float(probs[best_idx])
        # Return (cell, probability)
        return best_cell, best_prob

    def mine_probabilities(
        self, board, revealed: set, candidates: list, mines_known: set = None
    ) -> dict:
        """
        Return {cell: P(mine)} for every candidate.
        Used by BoardView to colour hint highlights.

        Parameters:
          - revealed: set of cells that have been revealed
          - candidates: list of cells to evaluate
          - mines_known: set of cells flagged/known as mines
        """
        if not self.trained or not candidates:
            return {}

        if mines_known is None:
            mines_known = set()

        feats = [
            extract_features(
                board, self.height, self.width, r, c, revealed, mines_known
            )
            for (r, c) in candidates
        ]
        X = np.array(feats, dtype=np.float32)
        probs = self.clf.predict_proba(X)[:, 1]
        return {cell: float(p) for cell, p in zip(candidates, probs)}

    # ── Feature importance ───────────────────────────────

    def log_feature_importance(self) -> None:
        pass  # disabled for cleaner output
