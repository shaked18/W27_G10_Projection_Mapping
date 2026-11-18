"""Low-latency camera capture.

The key to low latency with USB (UVC) cameras such as the Arducam UC-724 is to
never let frames queue up. OpenCV's VideoCapture keeps an internal buffer; if
the consumer is slower than the camera, every frame you read is already stale.

CameraStream solves this by reading from the device in a dedicated thread as
fast as the camera delivers and keeping ONLY the newest frame. The render loop
waits for the next new frame and never sees an old one.
"""

from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


def default_backend() -> int:
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Windows":
        # DirectShow usually has lower latency than MSMF for UVC cameras.
        return cv2.CAP_DSHOW
    return cv2.CAP_V4L2


def list_cameras(max_index: int = 8) -> list[tuple[int, int, int]]:
    """Probe camera indices; returns (index, width, height) for each that opens."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, default_backend())
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                found.append((i, frame.shape[1], frame.shape[0]))
        cap.release()
    return found


@dataclass
class Frame:
    seq: int            # increasing frame number
    image: np.ndarray   # BGR image
    t_capture: float    # time.perf_counter() when the frame left the driver


class CameraStream:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 60, mjpg: bool = True):
        self.index = index
        self.req_width = width
        self.req_height = height
        self.req_fps = fps
        self.mjpg = mjpg

        self.cap: cv2.VideoCapture | None = None
        self.width = 0
        self.height = 0

        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._seq = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self.capture_fps = 0.0
        self.error: str | None = None

    # ------------------------------------------------------------------ setup
    def start(self) -> "CameraStream":
        cap = cv2.VideoCapture(self.index, default_backend())
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index)  # let OpenCV pick a backend
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.index}. "
                "Run with --list-cameras to see available devices.")

        # Order matters on V4L2: set the pixel format before the resolution.
        # MJPG lets USB 2.0 cameras run high resolutions at full frame rate.
        if self.mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
        cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ignored by some backends; the thread handles it

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Camera {self.index} opened but returned no frames.")
        self.height, self.width = frame.shape[:2]
        self.cap = cap

        self._publish(frame)
        self._running = True
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return self

    # ---------------------------------------------------------------- capture
    def _publish(self, image: np.ndarray) -> None:
        t = time.perf_counter()
        with self._cond:
            self._seq += 1
            self._latest = Frame(self._seq, image, t)
            self._cond.notify_all()

    def _run(self) -> None:
        failures = 0
        last_t = time.perf_counter()
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 100:
                    self.error = "Camera stopped delivering frames (disconnected?)"
                    self._running = False
                    with self._cond:
                        self._cond.notify_all()
                    break
                time.sleep(0.005)
                continue
            failures = 0
            self._publish(frame)

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                inst = 1.0 / dt
                self.capture_fps = inst if self.capture_fps == 0 else 0.9 * self.capture_fps + 0.1 * inst

    # ----------------------------------------------------------------- access
    def wait_newer(self, last_seq: int, timeout: float = 0.05) -> Frame | None:
        """Block until a frame newer than `last_seq` exists (or timeout).

        Returns the newest frame, skipping any intermediate ones, or None.
        """
        with self._cond:
            if self._latest is None or self._latest.seq <= last_seq:
                self._cond.wait(timeout)
            if self._latest is not None and self._latest.seq > last_seq:
                return self._latest
        return None

    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
