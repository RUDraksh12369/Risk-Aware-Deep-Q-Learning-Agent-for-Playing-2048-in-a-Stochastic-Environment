"""
Pygame demo UI for presenting the 2048 agent.

This module adapts the visual layout/event-loop style from
https://github.com/arbelamram/pygame-2048 (MIT License) while using this
project's Game2048 engine as the only source of game rules. It is meant for
interactive demos and agent playback only; training and automated tests should
continue to use game/game.py and game/environment.py directly.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

import numpy as np
import pygame

from game.environment import encode_board
from game.game import ACTION_NAMES, Action, Game2048
from mcts.agent import MCTS2048Agent, MCTSConfig

FPS = 60
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 760
BOARD_SIZE_PX = 600
GRID_SIZE = 4
PADDING = 20
HEADER_HEIGHT = WINDOW_HEIGHT - BOARD_SIZE_PX
SIDEBAR_X = PADDING + BOARD_SIZE_PX + 30
SIDEBAR_WIDTH = WINDOW_WIDTH - SIDEBAR_X - PADDING
CELL_GAP = 10
CELL_SIZE = (BOARD_SIZE_PX - (CELL_GAP * (GRID_SIZE + 1))) // GRID_SIZE
MIN_DELAY_MS = 40
MAX_DELAY_MS = 1000

BACKGROUND_COLOR = (250, 248, 239)
BOARD_COLOR = (187, 173, 160)
EMPTY_CELL_COLOR = (205, 193, 180)
TEXT_COLOR = (119, 110, 101)
LIGHT_TEXT_COLOR = (249, 246, 242)
SUBTLE_TEXT_COLOR = (143, 122, 102)
PANEL_COLOR = (238, 228, 218)
BUTTON_COLOR = (143, 122, 102)
BUTTON_HOVER_COLOR = (119, 110, 101)

TILE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

KEY_ACTIONS = {
    pygame.K_UP: Action.UP,
    pygame.K_w: Action.UP,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_s: Action.DOWN,
    pygame.K_LEFT: Action.LEFT,
    pygame.K_a: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_d: Action.RIGHT,
}


class AgentLike(Protocol):
    def __call__(self, game: Game2048) -> Action | int:
        ...


@dataclass
class DemoConfig:
    mode: str
    seed: Optional[int]
    delay_ms: int
    agent: Optional[AgentLike] = None
    mcts_simulations: int = 160
    mcts_rollout_depth: int = 36


def choose_random_action(game: Game2048) -> Action:
    legal = game.legal_actions()
    return random.choice(legal)


def choose_heuristic_action(game: Game2048) -> Action:
    """Small deterministic baseline for demos until a trained agent is loaded."""
    legal = game.legal_actions()
    best_action = legal[0]
    best_key = None
    for action in legal:
        afterstate, gained, _ = game.simulate_move(action)
        empty_cells = int(np.sum(afterstate == 0))
        max_tile = int(afterstate.max())
        corner_bonus = int(max_tile in afterstate[[0, 0, -1, -1], [0, -1, 0, -1]])
        monotonic_bonus = _monotonicity_score(afterstate)
        key = (gained, empty_cells, corner_bonus, monotonic_bonus)
        if best_key is None or key > best_key:
            best_key = key
            best_action = action
    return best_action


def _monotonicity_score(board: np.ndarray) -> int:
    score = 0
    for row in board:
        score += int(np.sum(row[:-1] >= row[1:]))
    for col in board.T:
        score += int(np.sum(col[:-1] >= col[1:]))
    return score


def load_agent(spec: str) -> AgentLike:
    """Load an agent chooser from 'package.module:callable_name'."""
    if ":" not in spec:
        raise ValueError("Agent spec must look like 'package.module:callable_name'.")
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    agent = getattr(module, attr_name)
    if not callable(agent):
        raise TypeError(f"{spec!r} is not callable.")
    return agent


def normalize_action(action: Action | int) -> Action:
    try:
        return Action(action)
    except ValueError as exc:
        raise ValueError(f"Agent returned invalid action {action!r}.") from exc


class Pygame2048Demo:
    def __init__(self, config: DemoConfig):
        pygame.init()
        pygame.display.set_caption("2048 MCTS Demo")
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.config = config
        self.game = Game2048(seed=config.seed)
        self.mcts_agent = MCTS2048Agent(
            MCTSConfig(
                simulations=config.mcts_simulations,
                rollout_depth=config.mcts_rollout_depth,
                seed=config.seed,
            )
        )
        self.title_font = pygame.font.SysFont("arial", 52, bold=True)
        self.score_font = pygame.font.SysFont("arial", 28, bold=True)
        self.small_font = pygame.font.SysFont("arial", 20)
        self.tiny_font = pygame.font.SysFont("arial", 16)
        self.tile_font = pygame.font.SysFont("arial", 46, bold=True)
        self.message_font = pygame.font.SysFont("arial", 44, bold=True)
        self.last_action: Optional[Action] = None
        self.last_reward = 0
        self.last_step_at = 0
        self.paused = config.mode == "human"
        self.dragging_speed = False
        self.speed_track = pygame.Rect(SIDEBAR_X, 277, SIDEBAR_WIDTH, 10)
        self.slower_button = pygame.Rect(SIDEBAR_X, 315, 74, 38)
        self.faster_button = pygame.Rect(SIDEBAR_X + SIDEBAR_WIDTH - 74, 315, 74, 38)

    def run(self) -> None:
        running = True
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_keydown(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_mouse_down(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging_speed = False
                elif event.type == pygame.MOUSEMOTION and self.dragging_speed:
                    self._set_delay_from_mouse(event.pos[0])

            if self._should_autoplay():
                self._step_autoplay()

            self._draw()

        pygame.quit()

    def _handle_keydown(self, key: int) -> bool:
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if key == pygame.K_r:
            self._reset()
        elif key == pygame.K_SPACE and self.config.mode != "human":
            self.paused = not self.paused
        elif key in (pygame.K_EQUALS, pygame.K_PLUS):
            self._adjust_delay(-40)
        elif key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
            self._adjust_delay(40)
        elif self.config.mode == "human" and key in KEY_ACTIONS:
            self._step(KEY_ACTIONS[key])
        return True

    def _handle_mouse_down(self, position: tuple[int, int]) -> None:
        if self.slower_button.collidepoint(position):
            self._adjust_delay(60)
        elif self.faster_button.collidepoint(position):
            self._adjust_delay(-60)
        elif self.speed_track.inflate(0, 18).collidepoint(position):
            self.dragging_speed = True
            self._set_delay_from_mouse(position[0])

    def _adjust_delay(self, delta_ms: int) -> None:
        self.config.delay_ms = max(MIN_DELAY_MS, min(MAX_DELAY_MS, self.config.delay_ms + delta_ms))

    def _set_delay_from_mouse(self, x_position: int) -> None:
        ratio = (x_position - self.speed_track.left) / self.speed_track.width
        ratio = max(0.0, min(1.0, ratio))
        self.config.delay_ms = int(MAX_DELAY_MS - ratio * (MAX_DELAY_MS - MIN_DELAY_MS))

    def _reset(self) -> None:
        self.game.reset()
        self.last_action = None
        self.last_reward = 0
        self.last_step_at = 0

    def _should_autoplay(self) -> bool:
        if self.config.mode == "human" or self.paused or self.game.done:
            return False
        now = pygame.time.get_ticks()
        return now - self.last_step_at >= self.config.delay_ms

    def _step_autoplay(self) -> None:
        chooser = {
            "random": choose_random_action,
            "heuristic": choose_heuristic_action,
            "mcts": self.mcts_agent.choose_action,
            "agent": self._choose_agent_action,
        }[self.config.mode]
        self._step(chooser(self.game))

    def _choose_agent_action(self, game: Game2048) -> Action:
        if self.config.agent is None:
            raise RuntimeError("Agent mode requires --agent package.module:callable_name.")
        try:
            return normalize_action(self.config.agent(game))
        except TypeError:
            encoded = encode_board(game.board)
            return normalize_action(self.config.agent(encoded))  # type: ignore[arg-type]

    def _step(self, action: Action | int) -> None:
        action = normalize_action(action)
        _, reward, _, info = self.game.step(action)
        if info["valid"]:
            self.last_action = action
            self.last_reward = reward
            self.last_step_at = pygame.time.get_ticks()

    def _draw(self) -> None:
        self.window.fill(BACKGROUND_COLOR)
        self._draw_header()
        self._draw_board()
        self._draw_sidebar()
        if self.game.done:
            self._draw_overlay("Game Over", "Press R to restart")
        pygame.display.flip()

    def _draw_header(self) -> None:
        self._draw_text("2048", self.title_font, TEXT_COLOR, (PADDING, 22))
        self._draw_text(
            f"{self.config.mode.title()} demo",
            self.small_font,
            SUBTLE_TEXT_COLOR,
            (PADDING + 4, 82),
        )

        score_rect = pygame.Rect(SIDEBAR_X, 25, 105, 70)
        best_rect = pygame.Rect(SIDEBAR_X + 125, 25, 105, 70)
        self._draw_stat_box(score_rect, "Score", str(self.game.score))
        self._draw_stat_box(best_rect, "Max", str(self.game.max_tile()))

        status = self._status_text()
        self._draw_text(status, self.small_font, SUBTLE_TEXT_COLOR, (PADDING, 118))

    def _status_text(self) -> str:
        if self.config.mode == "human":
            return "Arrow keys/WASD move. R resets. Q/Esc quits."
        state = "paused" if self.paused else "running"
        last = ACTION_NAMES[self.last_action] if self.last_action is not None else "none"
        return f"Autoplay {state}. Last: {last}, reward {self.last_reward}."

    def _draw_stat_box(self, rect: pygame.Rect, label: str, value: str) -> None:
        pygame.draw.rect(self.window, BUTTON_COLOR, rect, border_radius=6)
        label_surf = self.small_font.render(label.upper(), True, PANEL_COLOR)
        value_surf = self.score_font.render(value, True, LIGHT_TEXT_COLOR)
        self.window.blit(label_surf, label_surf.get_rect(center=(rect.centerx, rect.y + 20)))
        self.window.blit(value_surf, value_surf.get_rect(center=(rect.centerx, rect.y + 48)))

    def _draw_board(self) -> None:
        board_rect = pygame.Rect(PADDING, HEADER_HEIGHT, BOARD_SIZE_PX, BOARD_SIZE_PX)
        pygame.draw.rect(self.window, BOARD_COLOR, board_rect, border_radius=8)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                value = int(self.game.board[row, col])
                x = board_rect.x + CELL_GAP + col * (CELL_SIZE + CELL_GAP)
                y = board_rect.y + CELL_GAP + row * (CELL_SIZE + CELL_GAP)
                self._draw_tile(value, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))

    def _draw_tile(self, value: int, rect: pygame.Rect) -> None:
        color = TILE_COLORS.get(value, (60, 58, 50)) if value else EMPTY_CELL_COLOR
        pygame.draw.rect(self.window, color, rect, border_radius=6)
        if not value:
            return

        font = self.tile_font
        if value >= 1024:
            font = pygame.font.SysFont("arial", 36, bold=True)
        elif value >= 128:
            font = pygame.font.SysFont("arial", 40, bold=True)

        text_color = TEXT_COLOR if value <= 4 else LIGHT_TEXT_COLOR
        tile_text = font.render(str(value), True, text_color)
        self.window.blit(tile_text, tile_text.get_rect(center=rect.center))

    def _draw_sidebar(self) -> None:
        mouse_pos = pygame.mouse.get_pos()
        self._draw_text("Controls", self.score_font, TEXT_COLOR, (SIDEBAR_X, 138))
        mode = "Monte Carlo Tree Search" if self.config.mode == "mcts" else self.config.mode.title()
        self._draw_text(mode, self.small_font, SUBTLE_TEXT_COLOR, (SIDEBAR_X, 176))

        if self.config.mode == "mcts":
            self._draw_text(
                f"{self.config.mcts_simulations} sims/move",
                self.tiny_font,
                SUBTLE_TEXT_COLOR,
                (SIDEBAR_X, 205),
            )
            self._draw_text(
                f"rollout depth {self.config.mcts_rollout_depth}",
                self.tiny_font,
                SUBTLE_TEXT_COLOR,
                (SIDEBAR_X, 228),
            )

        self._draw_text("Speed", self.score_font, TEXT_COLOR, (SIDEBAR_X, 245))
        pygame.draw.rect(self.window, EMPTY_CELL_COLOR, self.speed_track, border_radius=5)
        speed_ratio = (MAX_DELAY_MS - self.config.delay_ms) / (MAX_DELAY_MS - MIN_DELAY_MS)
        knob_x = int(self.speed_track.left + speed_ratio * self.speed_track.width)
        pygame.draw.circle(self.window, BUTTON_COLOR, (knob_x, self.speed_track.centery), 10)
        self._draw_text(f"{self.config.delay_ms} ms", self.small_font, SUBTLE_TEXT_COLOR, (SIDEBAR_X, 292))

        self._draw_button(self.slower_button, "Slower", self.slower_button.collidepoint(mouse_pos))
        self._draw_button(self.faster_button, "Faster", self.faster_button.collidepoint(mouse_pos))

        lines = [
            "Space pause/resume",
            "R restart",
            "+/- adjust speed",
            "Q/Esc quit",
        ]
        y = 388
        for line in lines:
            self._draw_text(line, self.small_font, SUBTLE_TEXT_COLOR, (SIDEBAR_X, y))
            y += 30

    def _draw_button(self, rect: pygame.Rect, label: str, hovered: bool) -> None:
        color = BUTTON_HOVER_COLOR if hovered else BUTTON_COLOR
        pygame.draw.rect(self.window, color, rect, border_radius=6)
        text = self.small_font.render(label, True, LIGHT_TEXT_COLOR)
        self.window.blit(text, text.get_rect(center=rect.center))

    def _draw_overlay(self, title: str, subtitle: str) -> None:
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((250, 248, 239, 180))
        self.window.blit(overlay, (0, 0))
        title_surf = self.message_font.render(title, True, TEXT_COLOR)
        subtitle_surf = self.score_font.render(subtitle, True, SUBTLE_TEXT_COLOR)
        center_x = PADDING + BOARD_SIZE_PX // 2
        center_y = HEADER_HEIGHT + BOARD_SIZE_PX // 2
        self.window.blit(title_surf, title_surf.get_rect(center=(center_x, center_y - 28)))
        self.window.blit(subtitle_surf, subtitle_surf.get_rect(center=(center_x, center_y + 28)))

    def _draw_text(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        position: tuple[int, int],
    ) -> None:
        surf = font.render(text, True, color)
        self.window.blit(surf, position)


def parse_args(argv: Optional[Iterable[str]] = None) -> DemoConfig:
    parser = argparse.ArgumentParser(description="Run the demo-only pygame 2048 UI.")
    parser.add_argument(
        "--mode",
        choices=("human", "random", "heuristic", "mcts", "agent"),
        default="human",
        help="Demo control mode. Training/testing code is not used.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for repeatable demos.")
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=220,
        help="Autoplay delay between valid moves in milliseconds.",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=160,
        help="MCTS simulations per move for --mode mcts.",
    )
    parser.add_argument(
        "--mcts-rollout-depth",
        type=int,
        default=36,
        help="Maximum rollout depth for --mode mcts.",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="For --mode agent, load a chooser as 'package.module:callable_name'.",
    )
    args = parser.parse_args(argv)

    agent = load_agent(args.agent) if args.agent else None
    if args.mode == "agent" and agent is None:
        parser.error("--mode agent requires --agent package.module:callable_name")

    return DemoConfig(
        mode=args.mode,
        seed=args.seed,
        delay_ms=max(MIN_DELAY_MS, min(MAX_DELAY_MS, args.delay_ms)),
        agent=agent,
        mcts_simulations=args.mcts_simulations,
        mcts_rollout_depth=args.mcts_rollout_depth,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    config = parse_args(argv)
    demo = Pygame2048Demo(config)
    demo.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
