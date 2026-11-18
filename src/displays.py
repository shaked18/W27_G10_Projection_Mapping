"""Finding the projector display and covering it with the game window.

The projector must be an *extended* display (System Settings > Displays, not
"Mirror"), otherwise it can only ever show a copy of the laptop screen.

Displays are found with built-in OS APIs (no extra packages on Windows/Linux):
  * Windows: EnumDisplayMonitors via ctypes (process made DPI-aware so the
    coordinates are real pixels even with 125%/150% display scaling)
  * Linux:   xrandr --listmonitors
  * macOS:   AppKit (pip install pyobjc-framework-Cocoa)

Windows/Linux: the game window is moved onto the projector and made fullscreen
(OpenCV fullscreens on the monitor the window is on). macOS: the window is made
borderless and placed exactly over the projector's screen above the menu bar,
because OpenCV's fullscreen there goes to whichever screen has keyboard focus.
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

IS_MAC = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"


def _make_dpi_aware() -> None:
    """Windows: use real pixel coordinates for windows and monitors.

    Must run before any window is created, so it runs at import time.
    """
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


if IS_WINDOWS:
    _make_dpi_aware()


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


def _detect_windows() -> list[Screen]:
    import ctypes
    from ctypes import wintypes

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

    user32 = ctypes.windll.user32
    out: list[Screen] = []
    proc_type = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
                                   ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

    def callback(hmon, hdc, rect, lparam):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW(hmon, ctypes.byref(info))
        r = info.rcMonitor
        out.append(Screen(r.left, r.top, r.right - r.left, r.bottom - r.top,
                          bool(info.dwFlags & 1)))   # MONITORINFOF_PRIMARY
        return 1

    user32.EnumDisplayMonitors(None, None, proc_type(callback), 0)
    if not out:
        raise RuntimeError("no monitors")
    return out


def _detect_xrandr() -> list[Screen]:
    # Lines look like: " 1: +HDMI-1 1280/700x720/390+1920+0  HDMI-1"
    text = subprocess.run(["xrandr", "--listmonitors"], capture_output=True,
                          text=True, timeout=5, check=True).stdout
    out = []
    for line in text.splitlines()[1:]:
        m = re.search(r"(\d+)/\d+x(\d+)/\d+\+(-?\d+)\+(-?\d+)", line)
        if m:
            w, h, x, y = map(int, m.groups())
            out.append(Screen(x, y, w, h, "*" in line.split()[1]))
    if not out:
        raise RuntimeError("no monitors")
    if not any(s.primary for s in out):
        out[0].primary = True
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
    """The projector is assumed to be the first non-primary display.

    Prints why when it cannot be found, since that is the usual reason the
    projector ends up showing the whole desktop instead of only the game.
    """
    if IS_MAC:
        detectors = [_detect_macos]
    elif IS_WINDOWS:
        detectors = [_detect_windows]
    else:
        detectors = [_detect_xrandr]
    detectors.append(_detect_screeninfo)
    screens: list[Screen] = []
    for detect in detectors:
        try:
            screens = detect()
            break
        except Exception:
            continue
    else:
        pkg = "pyobjc-framework-Cocoa" if IS_MAC else "screeninfo"
        print(f"Can't list displays: run  pip install {pkg}  (or pass --projector-pos XxY).")
        return None

    for s in screens:
        print(f"display: {s.width}x{s.height} at ({s.x},{s.y})" + (" [main]" if s.primary else ""))
    for s in screens:
        if not s.primary:
            return s

    if IS_MAC:
        if mirroring_on():
            print("The projector MIRRORS the screen. Fix: System Settings > Displays > "
                  "projector > 'Use as: Extended display'.")
        else:
            print("Only one display found - is the projector connected?")
    elif IS_WINDOWS:
        print("Only one display found. If the projector is connected it is DUPLICATING the "
              "screen.\n  Fix: press Win+P and choose 'Extend', then restart this app.")
    else:
        print("Only one display found. If the projector is connected it is mirroring the "
              "screen.\n  Fix: in Display settings turn off 'Mirror displays' (or run "
              "xrandr --output <projector> --right-of <laptop>), then restart this app.")
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
