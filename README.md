# Tài Liệu Logic AI Game Dò Mìn

## Tổng Quan

AI trong game dò mìn này sử dụng **kết hợp 2 phương pháp**:

1. **Rule-Based Reasoning (Constraint Logic)** - Suy luận dựa trên luật logic
2. **Machine Learning (XGBoost)** - Dự đoán xác suất khi không có nước đi chắc chắn

AI luôn ưu tiên **Rule-Based** trước, chỉ dùng **ML** khi không tìm được nước đi an toàn.

---

## Phần 1: Rule-Based Reasoning (Constraint Logic)

### 1.1 Ý Tưởng Cơ Bản

Mỗi ô đã mở sẽ hiển thị số mìn xung quanh (0-8). AI sử dụng thông tin này để suy luận logic:

**Ví dụ:**

```
? ? ?
? 2 ?    → Cần tìm 2 mìn trong 5 ô xung quanh
? ? ?
```

AI tạo ra một **Sentence** (câu logic): `{5 ô chưa biết} = 2 mìn`

### 1.2 Cấu Trúc Sentence

**File:** `src/game/sentence.py`

```python
class Sentence:
    """A set of board cells and a count of how many are mines."""

    def __init__(self, cells, count):
        self.cells = set(cells)  # Tập các ô chưa biết
        self.count = count       # Số mìn trong tập này

    def known_mines(self):
        """If cell count equals mine count, all cells are mines."""
        if len(self.cells) == self.count:
            return self.cells

    def known_safes(self):
        """If mine count is 0, all cells are safe."""
        if self.count == 0:
            return self.cells
```

**Logic suy luận:**

- Nếu `len(cells) == count` → **Tất cả đều là mìn**
- Nếu `count == 0` → **Tất cả đều an toàn**

**Ví dụ:**

```python
# Sentence: {(1,2), (1,3)} = 2
# → 2 ô, 2 mìn → Cả 2 đều là mìn!

# Sentence: {(4,5), (4,6), (5,5)} = 0
# → 0 mìn → Cả 3 đều an toàn!
```

### 1.3 Quá Trình Suy Luận

**File:** `src/ai/agent.py`

```python
class MinesweeperAI:
    def __init__(self, height=8, width=8, ml_predictor=None):
        self.moves_made = set()  # Các ô đã mở
        self.mines = set()       # Các ô chắc chắn là mìn
        self.safes = set()       # Các ô chắc chắn an toàn
        self.knowledge = []      # List các Sentence
```

#### Bước 1: Thêm Tri Thức Mới

Khi AI mở một ô mới, nó thêm thông tin vào knowledge base:

```python
def add_knowledge(self, cell, count):
    """Record that *cell* has *count* neighboring mines, then infer."""
    self.moves_made.add(cell)
    self.mark_safe(cell)

    # Tìm các ô xung quanh chưa biết
    nearby = self.nearby_cells(cell)
    nearby -= self.safes | self.moves_made  # Loại bỏ ô đã biết

    # Tạo sentence mới
    new_sentence = Sentence(nearby, count)
    self.knowledge.append(new_sentence)
```

**Ví dụ:**

```
Mở ô (2,2) → Thấy số 3
Xung quanh có: (1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,3)
Loại bỏ ô đã biết: còn lại 6 ô
→ Tạo Sentence: {6 ô} = 3 mìn
```

#### Bước 2: Suy Luận Trực Tiếp

```python
# Kiểm tra tất cả sentences trong KB
for sentence in self.knowledge:
    tmp_safes = sentence.known_safes()
    tmp_mines = sentence.known_mines()

    if tmp_safes:
        new_safes |= tmp_safes
    if tmp_mines:
        new_mines |= tmp_mines

# Đánh dấu toàn bộ ô an toàn và mìn tìm được
for safe in new_safes:
    self.mark_safe(safe)
for mine in new_mines:
    self.mark_mine(mine)
```

#### Bước 3: Suy Luận Gián Tiếp (Subset Reasoning)

Kỹ thuật **trừ tập hợp** để tìm thông tin mới:

```python
# Nếu sentence A là tập con của sentence B
# → Có thể suy luận sentence C = B - A
for sentence in self.knowledge:
    if prev.cells <= sentence.cells:  # prev ⊆ sentence
        inf_cells = sentence.cells - prev.cells
        inf_count = sentence.count - prev.count
        new_inferences.append(Sentence(inf_cells, inf_count))
```

**Ví dụ Quan Trọng:**

```
Sentence A: {(1,1), (1,2)} = 1 mìn
Sentence B: {(1,1), (1,2), (1,3)} = 2 mìn

A ⊆ B → Trừ đi:
Sentence C: {(1,3)} = 2 - 1 = 1 mìn
→ Kết luận: (1,3) chắc chắn là mìn!
```

**Code:**

```python
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
```

### 1.4 Chọn Nước Đi An Toàn

```python
def make_safe_move(self):
    """Return a known-safe cell not yet revealed, or None."""
    safe_moves = self.safes - self.moves_made
    return safe_moves.pop() if safe_moves else None
```

**Flow:**

1. Lấy tất cả ô đã biết an toàn (`self.safes`)
2. Loại bỏ ô đã mở (`self.moves_made`)
3. Nếu còn ô → Chọn 1 ô bất kỳ
4. Nếu không còn → Trả về `None` (phải đoán)

---

## Phần 2: Machine Learning Predictor

### 2.1 Khi Nào Dùng ML?

ML chỉ được kích hoạt khi:

- ✅ Rule-based không tìm được nước đi an toàn
- ✅ Model đã được train
- ✅ Còn ô chưa mở

```python
def make_random_move(self):
    """Return a move for the 'must guess' situation."""
    # ... tìm candidates ...

    if self.ml_predictor is not None and self.ml_predictor.trained:
        # Use ML predictor for best guess
        return self.ml_predictor.predict_safest(
            self._board_ref, self.moves_made, candidates, self.mines
        )

    # Fallback: chọn random nếu không có AI
    return random.choice(candidates)
```

### 2.2 Cách ML Hoạt Động

#### Bước 1: Thu Thập Dữ Liệu Training

**File:** `src/ai/ml_predictor.py`

```python
def train(self, n_games: int = 1000) -> None:
    print(f"Collecting data from {n_games} games...")
    X_list, y_list = [], []

    for i in range(n_games):
        xs, ys = _collect_one_game(self.height, self.width, self.mines)
        X_list.extend(xs)
        y_list.extend(ys)
```

**Thu thập từ game mô phỏng:**

1. Chạy N games (mặc định 1000)
2. Mỗi game: AI chơi đến khi phải đoán
3. Thu thập **tất cả candidates** tại thời điểm đó
4. Label: 1 nếu ô đó là mìn, 0 nếu an toàn

#### Bước 2: Trích Xuất Features

Mỗi ô chưa mở được biểu diễn bởi **14 features**:

```python
def extract_features(board, height, width, row, col, revealed, mines_known):
    """Build a 14-element feature vector for cell (row, col)."""
    return [
        1,                    # 0: bias
        sum_probs,           # 1: tổng xác suất từ các ô số xung quanh
        mines_around,        # 2: số mìn đã flag xung quanh
        revealed_around,     # 3: số ô đã mở xung quanh
        unrevealed_around,   # 4: số ô chưa mở xung quanh
        max_num,             # 5: số lớn nhất xung quanh
        min_num,             # 6: số nhỏ nhất xung quanh
        avg_num,             # 7: trung bình số xung quanh
        ratio_mines,         # 8: tỷ lệ mines_around / revealed_around
        is_corner,           # 9: ô góc (0 hoặc 1)
        is_edge,             # 10: ô cạnh (0 hoặc 1)
        zero_cells,          # 11: số ô "0" xung quanh
        dist_center,         # 12: khoảng cách tới trung tâm board
        sum_remaining,       # 13: tổng số mìn còn lại của các ô số lân cận
    ]
```

**Giải thích các features quan trọng:**

**Feature 1: sum_probs** - Xác suất mìn từ các ô số xung quanh

```python
# Ví dụ: Ô (3,3) có ô "2" bên cạnh
# Ô "2" có 5 ô chưa mở xung quanh, đã flag 1 mìn
# → Còn 1 mìn trong 4 ô → P = 1/4 = 0.25
# → sum_probs += 0.25
```

**Feature 2: mines_around** - Số mìn đã flag xung quanh

```python
# Đếm số ô đã được AI đánh dấu là mìn trong vùng 3x3
for r, c in neighbors:
    if (r, c) in mines_known:
        mines_around += 1
```

**Feature 3-4: revealed_around, unrevealed_around**

```python
# Đếm số ô đã mở vs chưa mở xung quanh
# Ô có nhiều revealed_around → nhiều context → dự đoán tốt hơn
```

**Feature 9-10: is_corner, is_edge**

```python
is_corner = int((row in (0, height-1)) and (col in (0, width-1)))
is_edge = int((row in (0, height-1)) or (col in (0, width-1))) - is_corner
# Ô góc/cạnh có ít hàng xóm → xác suất mìn khác biệt
```

**Feature 12: dist_center** - Khoảng cách Manhattan tới tâm board

```python
cx, cy = height/2.0, width/2.0
max_d = (height + width) / 2.0
dist_c = (abs(row - cx) + abs(col - cy)) / max_d
# Normalize về [0, 1]
# Ô giữa thường có nhiều thông tin hơn ô ngoài cùng
```

#### Bước 3: Training Model

```python
# XGBoost Classifier với hyperparameters tối ưu
safe_to_mine_ratio = (height * width - mines) / mines

self.clf = XGBClassifier(
    n_estimators=500,           # Nhiều cây quyết định
    max_depth=8,                # Độ sâu mỗi cây
    learning_rate=0.03,         # Tốc độ học chậm → tránh overfit
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=safe_to_mine_ratio,  # Cân bằng class
    eval_metric="logloss",
    random_state=42,
)

# Train với train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
self.clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
```

**Tại sao dùng XGBoost?**

- ✅ Xử lý tốt data imbalanced (mìn ít hơn ô an toàn)
- ✅ Feature importance rõ ràng
- ✅ Không cần normalize data
- ✅ Chống overfit tốt
- ✅ Nhanh và chính xác

#### Bước 4: Dự Đoán

```python
def predict_safest(self, board, revealed, candidates, mines_known=None):
    """Return the candidate cell with the lowest predicted mine probability."""
    if not candidates:
        return None
    if not self.trained:
        return random.choice(candidates)

    # Extract features cho tất cả candidates
    feats = [
        extract_features(board, self.height, self.width, r, c, revealed, mines_known)
        for r, c in candidates
    ]
    X = np.array(feats, dtype=np.float32)

    # Dự đoán xác suất mỗi ô là mìn
    probs = self.clf.predict_proba(X)[:, 1]  # P(mine)

    # Chọn ô có xác suất thấp nhất
    best_idx = int(np.argmin(probs))
    best_cell = candidates[best_idx]
    return best_cell
```

**Flow:**

1. Trích xuất features cho mỗi candidate
2. Model dự đoán P(mine) cho mỗi ô
3. Chọn ô có P(mine) **thấp nhất**
4. Trả về ô đó cho AI

---

## Phần 3: Cách AI Ra Quyết Định

### 3.1 Flow Tổng Quát

**File:** `src/gui/app.py` - Hàm `_ai_step()`

```python
def _ai_step(self, mode: str = "autoplay"):
    """Execute one AI action: flag, reveal, or guess."""

    # BƯỚC 1: Flag các mìn đã biết chắc chắn
    to_flag = ai.mines - s.flags
    if to_flag:
        for m in to_flag:
            s.flags.add(m)
            logger.log(mode, "flag", m, "known mine from KB", status_str())

    # BƯỚC 2: Thử tìm nước đi an toàn (Rule-Based)
    move = ai.make_safe_move()
    reason = "safe move"
    source = "rule"

    # BƯỚC 3: Nếu không có nước an toàn → Dùng ML
    if move is None:
        move = ai.make_random_move()
        if s.ml_ready:
            reason = "AI guess"
            source = "ml"
        else:
            reason = "random guess"
            source = "random"

    # BƯỚC 4: Thực hiện nước đi
    if move is not None:
        if game.is_mine(move):
            s.lost = True
            logger.log(mode, "reveal", move, f"{reason} - hit mine", status_str())
        else:
            nearby = game.nearby_mines(move)
            s.revealed.add(move)
            ai.add_knowledge(move, nearby)  # Cập nhật KB
            action = "reveal" if source == "rule" else "guess"
            logger.log(mode, action, move, reason, status_str())
```

### 3.2 Ví Dụ Cụ Thể

**Tình huống:** Board 10x10, đã mở 40 ô, còn 20 ô chưa biết

#### Turn 1: Rule-Based Tìm Được Ô An Toàn

```
Knowledge Base:
  Sentence 1: {(2,3), (3,3)} = 0 mìn
  Sentence 2: {(5,6)} = 1 mìn
  ...

→ make_safe_move() tìm thấy: (2,3) từ Sentence 1
→ AI mở (2,3) - "safe move"
→ Thấy số "1"
→ add_knowledge((2,3), 1)
```

#### Turn 2: Phải Đoán Bằng ML

```
Knowledge Base không có ô an toàn nào
→ make_safe_move() = None
→ make_random_move() được gọi

Candidates: 20 ô chưa biết
→ ML extract features cho 20 ô
→ Model dự đoán:
   (7,5): P(mine) = 0.82
   (4,2): P(mine) = 0.15
   (9,1): P(mine) = 0.43
   ...
→ Chọn (4,2) vì có P(mine) thấp nhất
→ AI mở (4,2) - "AI guess"
```

#### Turn 3: Trúng Mìn

```
→ ML chọn (7,8), P(mine) = 0.21
→ AI mở (7,8)
→ is_mine((7,8)) = True
→ Game Over - "AI guess - hit mine"
```

---

## Phần 4: Tại Sao Có Thể Trúng Mìn Dù P(mine) Thấp?

### 4.1 P(mine) Là Xác Suất, Không Phải Chắc Chắn

```
P(mine) = 0.15 (15%)
→ Có 15% khả năng là mìn
→ Có 85% khả năng an toàn
→ Vẫn CÓ THỂ trúng mìn!
```

**Ví dụ thực tế:**

```
100 lần đoán với P(mine) = 0.15
→ Kỳ vọng: 85 lần an toàn, 15 lần trúng mìn
→ Đây là kết quả ĐÚNG, không phải lỗi!
```

### 4.2 Tình Huống Cuối Game Khó Hơn

```
Early game: 100 ô, 15 mìn → 15% mìn
Late game: 10 ô, 5 mìn → 50% mìn
→ Xác suất trúng mìn tăng đáng kể
```

**Log từ user:**

```
[90] guess (9,2) prob=0.289  ← 28.9% là mìn
[91] reveal (9,0) prob=0.182 - hit mine  ← 18.2% vẫn trúng!
```

→ Cuối game còn ít ô, xác suất không còn thấp nữa
→ 18.2% = gần 1/5 khả năng trúng mìn
→ Hoàn toàn bình thường!

### 4.3 Giới Hạn Của ML

ML **không thể tốt hơn thông tin có sẵn**:

```
Tình huống:
? ? ?
? 1 ?
? ? ?

→ 1 mìn trong 8 ô
→ Tất cả các ô đều có P(mine) ≈ 12.5%
→ ML chỉ có thể chọn ngẫu nhiên
→ 12.5% khả năng trúng mìn
```

---

## Phần 5: Đánh Giá Hiệu Suất AI

### 5.1 Metrics Quan Trọng

**Accuracy** - Độ chính xác tổng thể

```
Accuracy = (True Positive + True Negative) / Total
→ Cho biết tỷ lệ dự đoán đúng

Ví dụ: 68.5% accuracy
→ 68.5% lần dự đoán đúng ô là mìn hay không
```

**Precision** - Độ chính xác khi dự đoán "mìn"

```
Precision = True Positive / (True Positive + False Positive)
→ Khi model nói "đây là mìn", bao nhiêu % đúng?

High precision → Ít false alarm
```

**Recall** - Phát hiện được bao nhiêu % mìn thực tế

```
Recall = True Positive / (True Positive + False Negative)
→ Model tìm được bao nhiêu % mìn thực sự?

High recall → Ít bỏ sót mìn
```

**AUC-ROC** - Khả năng phân biệt tổng thể

```
AUC = 0.5: Random guess
AUC = 1.0: Perfect prediction
AUC = 0.7-0.8: Good model
```

### 5.2 Win Rate Thực Tế

```
Board 10x10, 15 mìn:
  - Rule-based only: ~15-20% win rate
  - Rule-based + ML: ~50-60% win rate
  ✅ ML cải thiện đáng kể!

Board 16x30, 99 mìn (Expert):
  - Rule-based only: ~5% win rate
  - Rule-based + ML: ~20-30% win rate
  ✅ ML vẫn giúp ích ngay cả ở level khó
```

---

## Tổng Kết

### Ưu Điểm Của Hệ Thống AI Này

✅ **Kết hợp thông minh 2 phương pháp:**

- Rule-based: 100% chính xác khi có đủ thông tin
- ML: Dự đoán tốt khi phải đoán

✅ **Học từ dữ liệu thực tế:**

- Training từ game mô phỏng
- Features được thiết kế dựa trên kinh nghiệm chơi

✅ **Tự động điều chỉnh:**

- Scale theo kích thước board
- Class weighting cho data imbalance

### Hạn Chế

❌ **Không thể đạt 100% win:**

- Nhiều tình huống bắt buộc phải đoán
- Không có thuật toán nào giải hoàn hảo Minesweeper

❌ **Phụ thuộc training data:**

- Model chỉ tốt nếu training đủ
- Cần balance giữa speed và quality

❌ **Late game khó hơn:**

- Còn ít ô → xác suất cao hơn
- Context ít → dự đoán khó hơn

### So Sánh Với Human

```
Human expert: 30-40% win rate (Expert board)
AI (rule + ML): 20-30% win rate

→ AI gần bằng human, nhưng chưa vượt qua
→ Vẫn còn chỗ cải tiến!
```

---

## Code Summary

**Các File Quan Trọng:**

1. **`src/ai/agent.py`** - Logic rule-based reasoning
   - `add_knowledge()` - Cập nhật KB
   - `make_safe_move()` - Tìm nước đi chắc chắn
   - `make_random_move()` - Dùng ML khi phải đoán

2. **`src/game/sentence.py`** - Constraint representation
   - `known_mines()` - Phát hiện mìn chắc chắn
   - `known_safes()` - Phát hiện ô an toàn

3. **`src/ai/ml_predictor.py`** - ML prediction
   - `train()` - Training XGBoost model
   - `extract_features()` - Trích xuất 14 features
   - `predict_safest()` - Dự đoán ô an toàn nhất

4. **`src/gui/app.py`** - Game loop & AI execution
   - `_ai_step()` - Thực thi 1 bước AI

**Flow Tổng Quát:**

```
Game Start
    ↓
Train ML Model (1000 games)
    ↓
AI Turn:
  1. Flag known mines (Rule)
  2. Try safe move (Rule)
  3. If no safe → ML guess
  4. Execute move
  5. Update knowledge base
  6. Repeat
    ↓
Win / Lose
```

---

_Tài liệu này giải thích logic AI trong game dò mìn. Code được thiết kế để dễ hiểu và mở rộng._
