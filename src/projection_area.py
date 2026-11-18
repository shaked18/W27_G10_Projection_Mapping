"""Find the borders of the projected area in the camera image.

Simple classical CV, no learned model: the projection is the largest bright
region the camera sees (works best in a dim room while the projector shows a
bright / white image).

    gray -> blur -> Otsu threshold -> morphological close
         -> largest contour -> approximate to a quadrilateral

Detection runs on a downscaled copy of the frame (~1 ms), and the corners are
smoothed over time so the outline does not jitter. Once the area is found it
can be locked, so later projected content (e.g. a dark game background) does
not move it.
"""

from __future__ import annotations

import cv2
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return 4 points ordered top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()   # y - x
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)],
                       pts[np.argmax(s)], pts[np.argmax(d)]])


class ProjectionAreaDetector:
    def __init__(self, work_width: int = 320, min_area_ratio: float = 0.05,
                 smoothing: float = 0.6):
        self.work_width = work_width          # detection resolution (speed vs precision)
        self.min_area_ratio = min_area_ratio  # ignore blobs smaller than 5% of the frame
        self.smoothing = smoothing            # 0 = no smoothing, closer to 1 = steadier
        self.corners: np.ndarray | None = None   # 4x2 float32 in full-frame pixels
        self.locked = False
        self.mask: np.ndarray | None = None      # last binary mask (work resolution), for debugging

    def reset(self) -> None:
        self.corners = None
        self.locked = False

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        """Detect on this frame (unless locked) and return the current corners."""
        if self.locked:
            return self.corners
        found = self.detect(frame)
        if found is None:
            return self.corners  # keep the last good result through brief dropouts
        if self.corners is None:
            self.corners = found
        else:
            a = self.smoothing
            self.corners = a * self.corners + (1 - a) * found
        return self.corners

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        h, w = frame.shape[:2]
        scale = self.work_width / w
        small = cv2.resize(frame, (self.work_width, int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Close gaps caused by dark content inside the projection.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.mask = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < self.min_area_ratio * mask.size:
            return None

        hull = cv2.convexHull(biggest)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in (0.02, 0.04, 0.06, 0.08):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                quad = approx
                break
        if quad is None:  # not a clean quadrilateral: fall back to a rotated box
            quad = cv2.boxPoints(cv2.minAreaRect(hull))

        return order_corners(quad) / scale

    # -------------------------------------------------------------- drawing
    def draw(self, img: np.ndarray, dim_outside: bool = False) -> None:
        """Draw the detected border (in place) on a display copy of the frame."""
        if self.corners is None:
            return
        pts = self.corners.astype(np.int32)
        if dim_outside:
            inside = np.zeros(img.shape[:2], np.uint8)
            cv2.fillConvexPoly(inside, pts, 255)
            img[inside == 0] //= 3
        color = (0, 200, 255) if self.locked else (0, 255, 0)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (int(x), int(y)), 6, color, -1, cv2.LINE_AA)
            cv2.putText(img, str(i + 1), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
