"""Camera -> projector alignment via a 4-point homography.

Procedure: the projector shows a bright frame with numbered corners. In the
operator window, click the 4 projected corners in the order
1 top-left, 2 top-right, 3 bottom-right, 4 bottom-left (as seen by the camera).
That gives a homography H mapping camera pixels to projector pixels, so any
point the camera detects (an object outline, a hand) can be drawn at the exact
physical spot by the projector. The game will reuse the same H.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CORNER_LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]


class Calibration:
    def __init__(self, path: Path, projector_size: tuple[int, int]):
        self.path = path
        self.proj_w, self.proj_h = projector_size
        self.cam_points: np.ndarray | None = None   # 4x2 float32, camera pixels
        self.cam_size: tuple[int, int] | None = None
        self.H: np.ndarray | None = None

    @property
    def valid(self) -> bool:
        return self.H is not None

    def _projector_corners(self) -> np.ndarray:
        w, h = self.proj_w - 1, self.proj_h - 1
        return np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    def set_points(self, points: list[tuple[float, float]], cam_size: tuple[int, int]) -> None:
        self.cam_points = np.float32(points)
        self.cam_size = cam_size
        self.H = cv2.getPerspectiveTransform(self.cam_points, self._projector_corners())

    def clear(self) -> None:
        self.cam_points = None
        self.cam_size = None
        self.H = None

    def adapt_to_camera(self, cam_size: tuple[int, int]) -> None:
        """Rescale stored points if the camera resolution changed since saving."""
        if self.cam_points is None or self.cam_size == cam_size:
            return
        sx = cam_size[0] / self.cam_size[0]
        sy = cam_size[1] / self.cam_size[1]
        pts = [(x * sx, y * sy) for x, y in self.cam_points]
        self.set_points(pts, cam_size)

    def warp(self, frame: np.ndarray) -> np.ndarray:
        """Re-render the camera frame from the projector's point of view."""
        return cv2.warpPerspective(frame, self.H, (self.proj_w, self.proj_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)

    def camera_to_projector(self, pts: np.ndarray) -> np.ndarray:
        """Map Nx2 camera-pixel points to projector pixels."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    # ----------------------------------------------------------- persistence
    def save(self) -> None:
        if self.cam_points is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "camera_points": self.cam_points.tolist(),
            "camera_size": list(self.cam_size),
            "projector_size": [self.proj_w, self.proj_h],
            "homography": self.H.tolist(),
        }, indent=2))

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            data = json.loads(self.path.read_text())
            self.set_points([tuple(p) for p in data["camera_points"]],
                            tuple(data["camera_size"]))
            return True
        except (KeyError, ValueError, TypeError):
            return False


def calibration_pattern(width: int, height: int) -> np.ndarray:
    """White border + numbered corner targets, easy to spot in the camera view."""
    img = np.zeros((height, width, 3), np.uint8)
    t = max(4, min(width, height) // 80)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (255, 255, 255), t)
    r = max(12, min(width, height) // 25)
    corners = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    for i, (x, y) in enumerate(corners):
        cv2.circle(img, (x, y), r, (0, 255, 255), -1)
        tx = x + (r + 10 if x == 0 else -r - 40)
        ty = y + (r + 40 if y == 0 else -r - 10)
        cv2.putText(img, str(i + 1), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (255, 255, 255), 3, cv2.LINE_AA)
    msg = "CALIBRATION: click corners 1-4 in the operator window"
    cv2.putText(img, msg, (width // 2 - 380, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img
