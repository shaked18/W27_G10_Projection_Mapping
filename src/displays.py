"""Monitor detection and the two output windows (operator screen + projector)."""

from __future__ import annotations

import platform
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Screen:
    x: int
    y: int
    width: int
    height: int
    name: str
    primary: bool


def _detect_macos() -> list[Screen]:
    from AppKit import NSScreen  # pip install pyobjc-framework-Cocoa

    screens = NSScreen.screens()
    if not screens:
        return []
    # Cocoa uses a bottom-left origin; convert to top-left like every other OS.
    main_h = screens[0].frame().size.height
    out = []
    for i, s in enumerate(screens):
        f = s.frame()
        name = s.localizedName() if hasattr(s, "localizedName") else f"screen{i}"
        out.append(Screen(
            x=int(f.origin.x),
            y=int(main_h - (f.origin.y + f.size.height)),
            width=int(f.size.width),
            height=int(f.size.height),
            name=str(name),
            primary=(i == 0),
        ))
    return out


def _detect_screeninfo() -> list[Screen]:
    from screeninfo import get_monitors  # pip install screeninfo

    return [Screen(m.x, m.y, m.width, m.height, m.name or f"screen{i}", bool(m.is_primary))
            for i, m in enumerate(get_monitors())]


def detect_screens() -> list[Screen]:
    detectors = [_detect_macos] if platform.system() == "Darwin" else []
    detectors.append(_detect_screeninfo)
    for detect in detectors:
        try:
            screens = detect()
            if screens:
                return screens
        except Exception:
            continue
    return []


def pick_projector(screens: list[Screen]) -> Screen | None:
    """The projector is assumed to be the first non-primary display."""
    for s in screens:
        if not s.primary:
            return s
    return None


class ProjectorWindow:
    """A borderless window shown fullscreen on the projector display."""

    def __init__(self, name: str, screen: Screen | None, size: tuple[int, int]):
        self.name = name
        self.screen = screen
        self.width, self.height = size
        self.fullscreen = False
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(name, self.width, self.height)
        if screen is not None:
            # Fullscreen happens on whichever display the window is on,
            # so move it onto the projector first.
            cv2.moveWindow(name, screen.x, screen.y)
            self.set_fullscreen(True)
        else:
            cv2.moveWindow(name, 60, 60)

    def set_fullscreen(self, on: bool) -> None:
        self.fullscreen = on
        cv2.setWindowProperty(
            self.name, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)

    def toggle_fullscreen(self) -> None:
        self.set_fullscreen(not self.fullscreen)

    def show(self, image: np.ndarray) -> None:
        cv2.imshow(self.name, image)


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit an image into width x height while keeping its aspect ratio."""
    h, w = image.shape[:2]
    if (w, h) == (width, height):
        return image
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    if (nw, nh) == (width, height):
        return resized
    canvas = np.zeros((height, width, 3), dtype=image.dtype)
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas
