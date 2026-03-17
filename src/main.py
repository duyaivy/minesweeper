"""Entry point — train ML model, then launch the Minesweeper GUI."""

import os
import sys
import logging
import argparse

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Minesweeper Game with AI")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Skip ML training and use random moves (faster startup)",
    )
    args = parser.parse_args()

    setup_logging()

    print("\n" + "=" * 50)
    print("MINESWEEPER GAME WITH AI")
    print(f"Board: {HEIGHT}x{WIDTH}, Mines: {MINES}")
    print("=" * 50)

    if not args.random:
        # Default: Train ML model
        print("Training AI model...")
        predictor = MLPredictor(height=HEIGHT, width=WIDTH, mines=MINES)
        predictor.train(n_games=1000)
        predictor.log_feature_importance()
        GameState.set_ml_predictor(predictor)
        print("Done training! Starting game...\n")
    else:
        # --random flag: Skip training
        print("ML Training DISABLED (--random flag)")
        print("AI will use random moves when stuck")
        print("Starting game...\n")
        GameState.set_ml_predictor(None)

    app = App()
    app.run()


if __name__ == "__main__":
    main()
