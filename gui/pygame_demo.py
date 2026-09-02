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
from typing import Callable, Iterable, Optional, Protocol

import numpy as np
import pygame

from game.environment import encode_board
from game.game import ACTION_NAMES, Action, Game2048

FPS = 60
WINDOW_SIZE = 760
BOARD_SIZE_PX = 600
GRID_SIZE = 4
PADDING = 20
HEADER_HEIGHT = WINDOW_SIZE - BOARD_SIZE_PX
CELL_GAP = 10
CELL_SIZE = (BOARD_SIZE_PX - (CELL_GAP * (GRID_SIZE + 1))) // GRID_SIZE

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
        pygame.display.set_caption("2048 Risk-Aware RL Demo")
        self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()
        self.config = config
        self.game = Game2048(seed=config.seed)
        self.title_font = pygame.font.SysFont("arial", 52, bold=True)
        self.score_font = pygame.font.SysFont("arial", 28, bold=True)
        self.small_font = pygame.font.SysFont("arial", 20)
        self.tile_font = pygame.font.SysFont("arial", 46, bold=True)
        self.message_font = pygame.font.SysFont("arial", 44, bold=True)
        self.last_action: Optional[Action] = None
        self.last_reward = 0
        self.last_step_at = 0
        self.paused = config.mode == "human"

    def run(self) -> None:
        running = True
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_keydown(event.key)

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
        elif self.config.mode == "human" and key in KEY_ACTIONS:
            self._step(KEY_ACTIONS[key])
        return True

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

        score_rect = pygame.Rect(WINDOW_SIZE - 250, 25, 105, 70)
        best_rect = pygame.Rect(WINDOW_SIZE - 130, 25, 105, 70)
        self._draw_stat_box(score_rect, "Score", str(self.game.score))
        self._draw_stat_box(best_rect, "Max", str(self.game.max_tile()))

        status = self._status_text()
        self._draw_text(status, self.small_font, SUBTLE_TEXT_COLOR, (PADDING, 118))

    def _status_text(self) -> str:
        if self.config.mode == "human":
            return "Arrow keys/WASD move. R resets. Q/Esc quits."
        state = "paused" if self.paused else "running"
        last = ACTION_NAMES[self.last_action] if self.last_action is not None else "none"
        return f"Autoplay {state}. Last: {last}, reward {self.last_reward}. Space pauses."

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

    def _draw_overlay(self, title: str, subtitle: str) -> None:
        overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
        overlay.fill((250, 248, 239, 180))
        self.window.blit(overlay, (0, 0))
        title_surf = self.message_font.render(title, True, TEXT_COLOR)
        subtitle_surf = self.score_font.render(subtitle, True, SUBTLE_TEXT_COLOR)
        center_x = WINDOW_SIZE // 2
        center_y = WINDOW_SIZE // 2
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
        choices=("human", "random", "heuristic", "agent"),
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
        "--agent",
        default=None,
        help="For --mode agent, load a chooser as 'package.module:callable_name'.",
    )
    args = parser.parse_args(argv)

    agent = load_agent(args.agent) if args.agent else None
    if args.mode == "agent" and agent is None:
        parser.error("--mode agent requires --agent package.module:callable_name")

    return DemoConfig(mode=args.mode, seed=args.seed, delay_ms=args.delay_ms, agent=agent)


def main(argv: Optional[Iterable[str]] = None) -> int:
    config = parse_args(argv)
    demo = Pygame2048Demo(config)
    demo.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
