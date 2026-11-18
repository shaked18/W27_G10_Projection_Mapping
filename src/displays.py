"""Finding the projector display and covering it with the game window.

The projector must be an *extended* display (System Settings > Displays, not
"Mirror"), otherwise it can only ever show a copy of the laptop screen.

On macOS (with pyobjc-framework-Cocoa installed) the game window is made
borderless and placed exactly over the projector's screen, above the menu bar,
so the projector shows only the game. OpenCV's own fullscreen is not used for
this on macOS because it goes to whichever screen has keyboard focus.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

IS_MAC = platform.system() == "Darwin"


@dataclass
class Screen:
    x: int
    y: int
    width: int
    height: int
    primary: bool
    native: Any = field(default=None, repr=False)   # NSScreen on macOS


def _detect_macos() -> list[Screen]:
    from AppKit import NSScreen  # pip install pyobjc-framework-Cocoa

    screens = NSScreen.screens()
    main_h = screens[0].frame().size.height
    out = []
    for i, s in enumerate(screens):
        f = s.frame()
        # Cocoa has a bottom-left origin; convert to top-left.
        out.append(Screen(int(f.origin.x), int(main_h - (f.origin.y + f.size.height)),
                          int(f.size.width), int(f.size.height), i == 0, s))
    return out


def _detect_screeninfo() -> list[Screen]:
    from screeninfo import get_monitors  # optional: pip install screeninfo

    return [Screen(m.x, m.y, m.width, m.height, bool(m.is_primary)) for m in get_monitors()]


def mirroring_on() -> bool:
    """True if macOS reports a mirrored display (projector copies the laptop screen)."""
    if not IS_MAC:
        return False
    try:
        out = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "Mirror: On" in out


def find_projector() -> Screen | None:
    """The projector is assumed to be the first non-primary display, if detectable."""
    detectors = [_detect_macos] if IS_MAC else []
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
        self.screen = screen
        self.fullscreen = False
        self._shown = False
        self._saved = None  # (style mask, frame, level) before covering the projector on macOS
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, *size)
        cv2.moveWindow(name, screen.x if screen else 80, screen.y if screen else 80)
        self._want_fullscreen = screen is not None

    def show(self, image: np.ndarray) -> None:
        cv2.imshow(self.name, image)
        if not self._shown:
            self._shown = True
            if self._want_fullscreen:
                self.toggle_fullscreen()

    def toggle_fullscreen(self) -> None:
        on = not self.fullscreen
        native = self.screen.native if self.screen else None
        if IS_MAC and native is not None and self._mac_cover(native, on):
            self.fullscreen = on
            return
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)
        self.fullscreen = on

    def _mac_cover(self, ns_screen, on: bool) -> bool:
        """Make the window borderless and exactly cover `ns_screen` (or restore it)."""
        try:
            from AppKit import (NSApplication, NSMainMenuWindowLevel,
                                NSWindowStyleMaskBorderless)
        except ImportError:
            return False
        win = next((w for w in NSApplication.sharedApplication().windows()
                    if w.title() == self.name), None)
        if win is None:
            return False
        if on:
            self._saved = (win.styleMask(), win.frame(), win.level())
            win.setStyleMask_(NSWindowStyleMaskBorderless)
            win.setFrame_display_(ns_screen.frame(), True)
            win.setLevel_(NSMainMenuWindowLevel + 1)   # above the menu bar / dock
        elif self._saved is not None:
            style, frame, level = self._saved
            win.setStyleMask_(style)
            win.setFrame_display_(frame, True)
            win.setLevel_(level)
        win.orderFront_(None)
        return True
