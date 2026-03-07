"""Main application — event loop, input dispatch, and rendering."""

import sys
import time
import pygame
import logging

log = logging.getLogger("minesweeper")
from src.config import (
    WIN_W,
    WIN_H,
    FPS,
    MINES,
    DARK_BG,
    LIGHT_GRAY,
    WHITE,
    BTN_COLOR,
    BTN_HOVER,
    BTN_TEXT,
    AI_TICK_MS,
    AI_MAX_MOVES_PER_TURN,
    CLICK_DELAY,
    FONT_PATH,
    HEIGHT,
    WIDTH,
)
from src.game.state import GameState
from src.game.rules import evaluate_game_status, check_mine_hit
from src.gui.hud import HUD
from src.gui.board_view import BoardView
from src.gui.sidebar import Sidebar
from src.gui.widgets import draw_button
from src.utils.geometry import pixel_to_cell


AI_AUTO_EVENT = pygame.USEREVENT + 1


class App:
    """Top-level application controller."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Minesweeper AI")
        self.clock = pygame.time.Clock()

        self.fonts = {
            "small": pygame.font.Font(FONT_PATH, 20),
            "medium": pygame.font.Font(FONT_PATH, 28),
            "large": pygame.font.Font(FONT_PATH, 40),
            "tiny": pygame.font.Font(FONT_PATH, 16),
            "cell": pygame.font.Font(FONT_PATH, 22),
        }

        self.hud = HUD(self.fonts)
        self.board_view = BoardView(self.fonts)
        self.sidebar = Sidebar(self.fonts)

        self.state = GameState()
        self.show_instructions = True

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        self.state = GameState()
        pygame.time.set_timer(AI_AUTO_EVENT, 0)

    def _toggle_autoplay(self):
        s = self.state
        s.ai_autoplay = not s.ai_autoplay
        if s.ai_autoplay and not s.is_game_over():
            s.start_timer_if_needed()
            pygame.time.set_timer(AI_AUTO_EVENT, AI_TICK_MS)
        else:
            pygame.time.set_timer(AI_AUTO_EVENT, 0)

    def _run_ai_inference(self, state):
        """Run AI inference to find new safe/mine cells after manual flag."""
        ai = state.ai

        try:
            # Simple one-pass inference
            new_safes = set()
            new_mines = set()

            # Clean up empty sentences
            ai.knowledge = [sent for sent in ai.knowledge if len(sent.cells) > 0]

            # Check for direct inferences
            for sentence in ai.knowledge:
                if len(sentence.cells) == 0:
                    continue

                tmp_safes = sentence.known_safes()
                tmp_mines = sentence.known_mines()

                if isinstance(tmp_safes, set) and tmp_safes:
                    new_safes |= tmp_safes
                if isinstance(tmp_mines, set) and tmp_mines:
                    new_mines |= tmp_mines

            # Mark all newly found safe cells (không tự động flag mines)
            for safe in new_safes:
                ai.mark_safe(safe)
            for mine in new_mines:
                ai.mark_mine(mine)
                # Chỉ hiển thị hint màu cam, không tự động flag
        except Exception as e:
            # Tránh crash game nếu có lỗi inference
            import logging

            log = logging.getLogger("minesweeper")
            log.error(f"Inference error: {e}")

    # ------------------------------------------------------------------
    # AI actions (with logging)
    # ------------------------------------------------------------------

    def _ai_step(self, mode: str = "autoplay"):
        """
        Execute AI actions: flag known mines, then reveal safe moves.

        Cân bằng giữa logic và UX:
        - Autoplay: Reveal 2-3 safe moves/turn (vừa đủ để inference, không quá nhanh)
        - Hint: Reveal 1 move (gợi ý cho người chơi)
        """
        s = self.state
        ai = s.ai
        game = s.game
        logger = s.ai_logger
        status_str = lambda: "lose" if s.lost else ("win" if s.won else "playing")

        # 1) Flag ALL known mines
        to_flag = ai.mines - s.flags
        if to_flag:
            for m in to_flag:
                s.flags.add(m)
                logger.log(mode, "flag", m, "known mine from KB", status_str())

        # 2) Reveal safe moves (số lượng vừa phải để cân bằng logic vs UX)
        revealed_count = 0
        max_safe_reveals = AI_MAX_MOVES_PER_TURN if mode == "autoplay" else 1

        while revealed_count < max_safe_reveals:
            move = ai.make_safe_move()
            if move is None:
                break  # Không còn safe moves

            # Reveal safe move
            if game.is_mine(move):
                # Không nên xảy ra (safe move không thể là mine)
                log.error(f"BUG: Safe move {move} is actually a mine!")
                s.lost = True
                logger.log(mode, "reveal", move, "safe move - ERROR", status_str())
                self._stop_autoplay()
                return

            # BFS flood-fill: mở toàn vùng 0 nếu cần
            newly_opened = game.reveal_flood_fill(move, s.revealed, s.flags)
            # Cập nhật AI knowledge cho tất cả ô mới mở
            for opened_cell in newly_opened:
                count = game.nearby_mines(opened_cell)
                ai.add_knowledge(opened_cell, count)

            logger.log(mode, "reveal", move, "safe move from KB", status_str())
            revealed_count += 1

            # Check win condition
            if evaluate_game_status(s, "ai_step", move) != "playing":
                if s.won:
                    logger.log(mode, "none", None, "game won!", "win")
                self._stop_autoplay()
                return

        # 3) Nếu đã reveal ít nhất 1 safe move → không guess trong lượt này
        # Vì reveal có thể tạo ra knowledge mới → có thể có safe moves mới ở lượt sau
        if revealed_count > 0:
            return

        # 4) Không còn safe moves → Phải guess (ML/random)
        move = ai.make_random_move()
        if s.ml_ready:
            prob = getattr(ai, "_last_ml_prob", None)
            if prob is not None:
                remaining = len(
                    [
                        c
                        for c in [
                            (r, c) for r in range(ai.height) for c in range(ai.width)
                        ]
                        if c not in ai.moves_made and c not in ai.mines
                    ]
                )
                log.info(
                    f"Using ML: {remaining} cells left, chose {move} P(mine)={prob:.3f}"
                )
            reason = "ML guess"
            source = "ml"
        else:
            reason = "random guess"
            source = "random"

        if move is not None:
            if game.is_mine(move):
                s.lost = True
                logger.log(mode, "guess", move, f"{reason} - hit mine", status_str())
                self._stop_autoplay()
            else:
                # BFS flood-fill
                newly_opened = game.reveal_flood_fill(move, s.revealed, s.flags)
                for opened_cell in newly_opened:
                    count = game.nearby_mines(opened_cell)
                    ai.add_knowledge(opened_cell, count)
                logger.log(mode, "guess", move, reason, status_str())
        else:
            # Không còn moves nào
            s.flags = ai.mines.copy()
            logger.log(mode, "none", None, "no moves left", status_str())
            self._stop_autoplay()

        # Evaluate game status after move
        evaluate_game_status(s, "ai_step", move)
        if s.won:
            logger.log(mode, "none", None, "game won!", "win")
            self._stop_autoplay()

    def _stop_autoplay(self):
        self.state.ai_autoplay = False
        pygame.time.set_timer(AI_AUTO_EVENT, 0)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while True:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == AI_AUTO_EVENT:
                    s = self.state
                    if s.ai_autoplay and not s.is_game_over():
                        self._ai_step("autoplay")

            self.screen.fill(DARK_BG)

            if self.show_instructions:
                self._draw_instructions(mouse_pos)
            else:
                self._draw_game(mouse_pos)

            pygame.display.flip()
            self.clock.tick(FPS)

    # ------------------------------------------------------------------
    # Screens
    # ------------------------------------------------------------------

    def _draw_instructions(self, mouse_pos):
        f = self.fonts
        title = f["large"].render("Minesweeper AI", True, WHITE)
        self.screen.blit(title, title.get_rect(center=(WIN_W // 2, 120)))

        rules = [
            "Nhấp chuột để mở tất cả các ô an toàn không có mìn",
            "Khi bấm vào một ô không có mìn → ô sẽ hiện ra một con số.",
            "Con số trên ô cho biết có bao nhiêu quả mìn nằm trong 8 ô xung quanh ô đó.",
            "Nhấp chuột phải để đánh dấu mìn.",
            "Đánh dấu tất cả mìn để chiến thắng!",
            "",
            "Chơi thủ công hoặc nhờ AI gợi ý từng bước.",
        ]
        for i, rule in enumerate(rules):
            line = f["small"].render(rule, True, LIGHT_GRAY)
            self.screen.blit(line, line.get_rect(center=(WIN_W // 2, 220 + 32 * i)))

        btn = pygame.Rect(WIN_W // 2 - 120, WIN_H - 160, 240, 56)
        draw_button(self.screen, btn, "Bắt Đầu", f["medium"], mouse_pos)

        click, _, _ = pygame.mouse.get_pressed()
        if click == 1 and btn.collidepoint(mouse_pos):
            self.show_instructions = False
            time.sleep(0.25)

    def _draw_game(self, mouse_pos):
        s = self.state
        s.tick_timer()

        # HUD
        mines_remaining = MINES - len(s.flags)
        self.hud.draw(self.screen, mines_remaining, s.elapsed, mouse_pos)

        # Board
        self.board_view.draw(self.screen, s)
        self.board_view.draw_hints(self.screen, s)

        # Sidebar
        self.sidebar.draw(self.screen, s, mouse_pos)

        # Input
        self._handle_game_input(mouse_pos)

    def _handle_game_input(self, mouse_pos):
        s = self.state
        left, _, right = pygame.mouse.get_pressed()

        if right == 1 and not s.is_game_over():
            cell = pixel_to_cell(mouse_pos)
            if cell and cell not in s.revealed:
                if cell in s.flags:
                    s.flags.remove(cell)
                else:
                    s.flags.add(cell)
                # Check win after flag action
                evaluate_game_status(s, "flag", cell)
                time.sleep(CLICK_DELAY)

        elif left == 1:
            # Reset
            if self.hud.reset_rect.collidepoint(mouse_pos):
                self.reset()
                time.sleep(0.2)
                return

            # AI Move (single step)
            if (
                self.sidebar.ai_btn_rect.collidepoint(mouse_pos)
                and not s.is_game_over()
            ):
                s.start_timer_if_needed()
                self._ai_step("hint")
                time.sleep(CLICK_DELAY)
                return

            # Checkbox
            if self.sidebar.cb_rect.collidepoint(mouse_pos):
                self._toggle_autoplay()
                time.sleep(0.25)
                return

            # Manual cell click
            if not s.is_game_over():
                cell = pixel_to_cell(mouse_pos)
                if cell and cell not in s.flags and cell not in s.revealed:
                    s.start_timer_if_needed()
                    # Check if mine hit
                    if check_mine_hit(s, cell):
                        # Game already lost, status logged in check_mine_hit
                        pass
                    else:
                        # BFS flood-fill: mở toàn vùng 0 nếu cần
                        newly_opened = s.game.reveal_flood_fill(
                            cell, s.revealed, s.flags
                        )
                        # Cập nhật AI knowledge cho tất cả ô mới mở
                        for opened_cell in newly_opened:
                            count = s.game.nearby_mines(opened_cell)
                            s.ai.add_knowledge(opened_cell, count)
                        # Check win after reveal
                        evaluate_game_status(s, "reveal", cell)
                    time.sleep(CLICK_DELAY)
