"""Finding the projector display and showing a fullscreen window on it."""

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
    primary: bool


def _detect_macos() -> list[Screen]:
    from AppKit import NSScreen  # optional: pip install pyobjc-framework-Cocoa

    screens = NSScreen.screens()
    main_h = screens[0].frame().size.height
    out = []
    for i, s in enumerate(screens):
        f = s.frame()
        # Cocoa has a bottom-left origin; convert to top-left.
        out.append(Screen(int(f.origin.x), int(main_h - (f.origin.y + f.size.height)),
                          int(f.size.width), int(f.size.height), i == 0))
    return out


def _detect_screeninfo() -> list[Screen]:
    from screeninfo import get_monitors  # optional: pip install screeninfo

    return [Screen(m.x, m.y, m.width, m.height, bool(m.is_primary)) for m in get_monitors()]


def find_projector() -> Screen | None:
    """The projector is assumed to be the first non-primary display, if detectable."""
    detectors = [_detect_macos] if platform.system() == "Darwin" else []
    detectors.append(_detect_screeninfo)
    for detect in detectors:
        try:
            screens = detect()
        except Exception:
            continue
        for s in screens:
            if not s.primary:
                return s
        if screens:
            return None
    return None


class ProjectorWindow:
    def __init__(self, name: str, screen: Screen | None, size: tuple[int, int]):
        self.name = name
        self.fullscreen = False
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(name, *size)
        if screen is not None:
            # Fullscreen happens on the display the window is on, so move it there first.
            cv2.moveWindow(name, screen.x, screen.y)
            self.toggle_fullscreen()
        else:
            cv2.moveWindow(name, 80, 80)

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)

    def show(self, image: np.ndarray) -> None:
        cv2.imshow(self.name, image)
