"""A simple Snake game rendered with OpenCV, meant to be shown on the projector.

The game draws a bright border around the play field so the camera can keep
detecting the borders of the projected area (projection_area.py) while the
game background stays dark.
"""

from __future__ import annotations

import random
import time

import cv2
import numpy as np

UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)

BG = (20, 20, 20)
BORDER = (255, 255, 255)
SNAKE_HEAD = (80, 255, 120)
SNAKE_BODY = (40, 190, 80)
FOOD = (60, 60, 255)
TEXT = (255, 255, 255)


class SnakeGame:
    def __init__(self, width: int, height: int, cell: int = 40,
                 border: int = 12, start_speed: float = 7.0):
        self.width, self.height = width, height
        self.cell = cell
        self.border = border
        self.cols = (width - 2 * border) // cell
        self.rows = (height - 2 * border) // cell
        # Center the grid inside the border.
        self.ox = (width - self.cols * cell) // 2
        self.oy = (height - self.rows * cell) // 2
        self.start_speed = start_speed      # moves per second
        self.best = 0
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        cx, cy = self.cols // 2, self.rows // 2
        self.snake = [(cx - i, cy) for i in range(3)]   # head first
        self.direction = RIGHT
        self.pending: list[tuple[int, int]] = []        # buffered turns (max 2)
        self.score = 0
        self.speed = self.start_speed
        self.game_over = False
        self.paused = False
        self.last_step = time.perf_counter()
        self.food = self._spawn_food()

    def _spawn_food(self) -> tuple[int, int]:
        free = {(x, y) for x in range(self.cols) for y in range(self.rows)} - set(self.snake)
        return random.choice(sorted(free)) if free else self.snake[0]

    def turn(self, d: tuple[int, int]) -> None:
        last = self.pending[-1] if self.pending else self.direction
        # Ignore reversing into yourself and duplicate presses.
        if d != last and (d[0] + last[0], d[1] + last[1]) != (0, 0) and len(self.pending) < 2:
            self.pending.append(d)

    def toggle_pause(self) -> None:
        if not self.game_over:
            self.paused = not self.paused
            self.last_step = time.perf_counter()

    def update(self) -> bool:
        """Advance the game if a step is due. Returns True if the picture changed."""
        if self.game_over or self.paused:
            return False
        now = time.perf_counter()
        if now - self.last_step < 1.0 / self.speed:
            return False
        self.last_step = now

        if self.pending:
            self.direction = self.pending.pop(0)
        hx, hy = self.snake[0]
        head = (hx + self.direction[0], hy + self.direction[1])

        hits_wall = not (0 <= head[0] < self.cols and 0 <= head[1] < self.rows)
        grows = head == self.food
        body = self.snake if grows else self.snake[:-1]   # tail moves away unless growing
        if hits_wall or head in body:
            self.game_over = True
            self.best = max(self.best, self.score)
            return True

        self.snake.insert(0, head)
        if grows:
            self.score += 1
            self.speed = min(self.start_speed + self.score * 0.4, 20.0)
            self.food = self._spawn_food()
        else:
            self.snake.pop()
        return True

    # -------------------------------------------------------------- rendering
    def _cell_rect(self, x: int, y: int, pad: int = 2) -> tuple[tuple[int, int], tuple[int, int]]:
        x0, y0 = self.ox + x * self.cell, self.oy + y * self.cell
        return (x0 + pad, y0 + pad), (x0 + self.cell - pad, y0 + self.cell - pad)

    def render(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), BG, np.uint8)
        cv2.rectangle(img, (0, 0), (self.width - 1, self.height - 1), BORDER, self.border * 2)

        fx, fy = self.food
        c = self.cell
        cv2.circle(img, (self.ox + fx * c + c // 2, self.oy + fy * c + c // 2), c // 2 - 4, FOOD, -1, cv2.LINE_AA)
        for i, (x, y) in enumerate(self.snake):
            p0, p1 = self._cell_rect(x, y)
            cv2.rectangle(img, p0, p1, SNAKE_HEAD if i == 0 else SNAKE_BODY, -1)

        self._text(img, f"score {self.score}   best {self.best}", (self.border + 16, self.border + 36), 0.9)
        if self.game_over:
            self._center(img, "GAME OVER", 2.0, -30)
            self._center(img, "press Enter to play again", 0.9, 30)
        elif self.paused:
            self._center(img, "PAUSED", 2.0, 0)
        return img

    def _text(self, img, text, org, scale) -> None:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT, 2, cv2.LINE_AA)

    def _center(self, img, text, scale, dy) -> None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
        org = ((self.width - tw) // 2, (self.height + th) // 2 + dy)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT, 3, cv2.LINE_AA)
