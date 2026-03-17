# Minesweeper AI (Python)

A modern Minesweeper game built in Python with both **human gameplay** and an **AI assistant/player**.
You can play manually like classic Minesweeper, ask the AI for one-step hints, or enable autoplay so the AI solves the board over time.
The AI combines logical reasoning with a machine learning fallback to make smarter decisions when certainty is not possible.

## 1. Project introduction

This project demonstrates how a classic logic puzzle can be combined with practical AI techniques in a real game loop.

**Minesweeper** is a grid-based puzzle where each revealed number tells you how many mines exist in its 8 neighboring cells. The player wins by revealing all safe cells and avoiding mines.

AI is integrated to:

- show how rule-based reasoning can solve deterministic game states,
- reduce random guessing with probability-based predictions,
- provide an educational example of AI decision-making in games.

This project is designed for **learning, experimentation, and demonstration**:

- Learning: understand game state modeling, inference rules, and AI integration.
- Experimentation: tune board settings and ML training size.
- Demonstration: visualize AI behavior in a playable GUI.

## 2. Demo

![Minesweeper AI gameplay demo](assets/demo.gif)

_Demo: A full game session showing manual reveals, mine flagging, AI hint/autoplay actions, and end-state win/lose handling._

## 3. Key features

- Classic Minesweeper rules and grid mechanics.
- Left-click reveal, right-click flag/unflag workflow.
- Flood-fill reveal for zero-value cells.
- Real-time win/lose detection and game reset.
- AI knowledge base with logical inference from revealed clues.
- One-step AI hint mode (assistive play).
- AI autoplay mode (automated play loop).
- ML-based fallback move selection when logic cannot guarantee safety.
- Probability-aware AI guesses using a trained XGBoost model.
- GUI built with Pygame (board view, HUD, sidebar controls).
- Configurable board setup via central constants (height/width/mine count).

## 4. AI algorithm used

The AI in this project is a **hybrid strategy**:

1. **Rule-based logical inference** (primary)
2. **Probabilistic ML prediction** (fallback when forced to guess)

This is a strong fit for Minesweeper because the game naturally contains:

- deterministic logic (many safe/mine conclusions are provable), and
- uncertain states (some positions require probabilistic guessing).

### Idea

Use strict logic whenever possible, and only use learned probability when no guaranteed safe move exists.

- Logic layer = high precision, explainable decisions.
- ML layer = better-than-random decisions under uncertainty.

### How it works

#### A. Knowledge-based logical inference (Rule-based)

The AI stores constraints as sentences of the form:

`{set of hidden neighbor cells} = number of mines in that set`

When a cell is revealed with clue `count`, the AI:

1. Marks that cell as safe and visited.
2. Builds a sentence from its unknown neighbors.
3. Updates known mines/safes across all existing sentences.
4. Repeats inference until no new information appears.

Core deduction rules:

- If `count == 0`, all cells in the sentence are safe.
- If `len(cells) == count`, all cells in the sentence are mines.
- If sentence A is a subset of sentence B, infer a new sentence `B - A` with mine count difference.

This subset inference is critical for solving non-trivial board states.

#### B. Probabilistic fallback (ML with XGBoost)

If no safe move can be proven, the AI must guess. Instead of random guessing, it uses a trained model to estimate mine probability for candidate cells.

Pipeline:

1. Simulate many games to collect training samples.
2. Build feature vectors for candidate hidden cells (e.g., nearby revealed clues, local uncertainty, edge/corner context).
3. Train an `XGBoost` classifier to predict `P(cell is mine)`.
4. During gameplay, choose the candidate with **lowest predicted mine probability**.

### Example

Suppose the AI currently has:

- Sentence A: `{(2,3), (2,4)} = 1`
- Sentence B: `{(2,3), (2,4), (2,5)} = 2`

Because A is a subset of B, subtract A from B:

- Inferred sentence C: `{(2,5)} = 1`

So `(2,5)` is definitely a mine.

Then, if another clue implies:

- Sentence D: `{(1,5), (1,6)} = 0`

both `(1,5)` and `(1,6)` are guaranteed safe and can be revealed.

If no guaranteed-safe cells remain after all inferences, the AI evaluates border candidates with ML and picks the lowest-risk cell.

### Advantages

- Highly explainable behavior from the logic layer.
- Strong performance on deterministic board states.
- Lower guess risk than pure random fallback.
- Practical architecture for combining symbolic + statistical AI.

### Limitations

- Minesweeper still includes unavoidable uncertainty in some states.
- ML predictions depend on training quality and representativeness.
- Startup can be slower when training is enabled (default training run).
- AI is optimized for this project’s board setup and feature design, not guaranteed globally optimal.

## 5. Technologies used

- **Python 3**
- **Pygame** (graphical interface and event loop)
- **NumPy** (numerical arrays for ML features)
- **XGBoost** (mine-probability classifier)
- **scikit-learn** (train/validation split and metrics)
- Python standard libraries (`random`, `argparse`, `logging`, `time`, etc.)
- Custom modules for game logic, AI reasoning, GUI, and utilities

## 6. Installation and run

### Prerequisites

- Python 3.10+ recommended
- `pip` available in your environment

### Setup

```bash
# Clone your repository
git clone https://github.com/duyaivy/minesweeper.git
cd minesweeper

# (Optional) Create and activate virtual environment
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows Git Bash
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# Standard run (includes ML training before game starts)
python runner.py

# Faster startup: skip ML training and use random fallback when stuck
python runner.py --random
```

## 7. Project structure

```bash
minesweeper/
│── README.md
│── requirements.txt
│── runner.py
│── assets/
│   ├── demo.gif
│   ├── fonts/
│   └── images/
│── src/
│   ├── config.py
│   ├── main.py
│   ├── ai/
│   │   ├── agent.py
│   │   ├── logger.py
│   │   └── ml_predictor.py
│   ├── game/
│   │   ├── board.py
│   │   ├── rules.py
│   │   ├── sentence.py
│   │   └── state.py
│   ├── gui/
│   │   ├── app.py
│   │   ├── assets.py
│   │   ├── board_view.py
│   │   ├── hud.py
│   │   ├── sidebar.py
│   │   └── widgets.py
│   └── utils/
│       └── geometry.py
```

## Notes

- Default board configuration is currently set in `src/config.py`.
- If you want to tune AI behavior, start with:
  - inference flow in `src/ai/agent.py`
  - ML feature extraction and training settings in `src/ai/ml_predictor.py`

---

If you use this project for coursework or demos, you can extend it with additional difficulty presets, model persistence (save/load), and comparative benchmarks between logic-only vs hybrid AI gameplay.

### From ![duyaivy](https://github.com/duyaivy) with 💖
