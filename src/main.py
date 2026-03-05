"""Entry point — train ML model, then launch the Minesweeper GUI."""

import os
import sys
import logging

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

from src.config import HEIGHT, WIDTH, MINES, setup_logging
from src.game.state import GameState
from src.ai.ml_predictor import MLPredictor
from src.gui.app import App

log = logging.getLogger("minesweeper")


def main():
    setup_logging()

    print("\n" + "=" * 50)
    print("MINESWEEPER GAME WITH AI")
    print(f"Board: {HEIGHT}x{WIDTH}, Mines: {MINES}")
    print("=" * 50)
    print("Training AI model...")
    predictor = MLPredictor(height=HEIGHT, width=WIDTH, mines=MINES)
    predictor.train(n_games=1000)
    predictor.log_feature_importance()

    # Attach to GameState so every new game uses the same trained model
    GameState.set_ml_predictor(predictor)

    print("Done training! Starting game...\n")
    app = App()
    app.run()


if __name__ == "__main__":
    main()
