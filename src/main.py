"""Live, low-latency view of what the camera sees, shown on the computer screen.

The projector is not driven by this program: it simply shows the computer's
screen (macOS display mirroring or an extended desktop). The camera is pointed
at the projected area; this window lets you see exactly what it captures.

Run:   python src/main.py                 (camera 0)
       python src/main.py --list-cameras  (find the Arducam UC-724 index)
       python src/main.py --camera 1

The UI outlines the borders of the projected area that the camera detects
(see projection_area.py). For best detection, dim the room and project a
bright image, then press 'l' to lock the area.

Keys:
  d  toggle projection-area detection
  l  lock / unlock the detected area
  o  dim everything outside the projected area
  r  reset detection
  f  toggle fullscreen
  h  toggle HUD (fps / latency)
  s  save a snapshot
  q / Esc  quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from camera import CameraStream, list_cameras
from projection_area import ProjectionAreaDetector

WINDOW = "Camera"
ROOT = Path(__file__).resolve().parent


def parse_size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time camera view")
    p.add_argument("--camera", type=int, default=0, help="camera index (see --list-cameras)")
    p.add_argument("--list-cameras", action="store_true", help="list cameras and exit")
    p.add_argument("--resolution", type=parse_size, default=(1280, 720),
                   help="requested camera resolution WxH (default 1280x720)")
    p.add_argument("--fps", type=int, default=60, help="requested camera FPS")
    p.add_argument("--no-mjpg", action="store_true", help="don't request MJPG from the camera")
    return p.parse_args()


class App:
    def __init__(self, args: argparse.Namespace):
        self.cam = CameraStream(args.camera, *args.resolution, fps=args.fps,
                                mjpg=not args.no_mjpg).start()
        print(f"camera {args.camera}: {self.cam.width}x{self.cam.height}")

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, min(self.cam.width, 1280), min(self.cam.height, 720))
        self.fullscreen = False
        self.show_hud = True
        self.frame: np.ndarray | None = None
        self.render_fps = 0.0
        self.latency_ms = 0.0
        self.detector = ProjectionAreaDetector()
        self.detect_on = True
        self.dim_outside = False

    def render(self, frame: np.ndarray) -> np.ndarray:
        if self.detect_on:
            self.detector.update(frame)
        if not (self.detect_on or self.show_hud):
            return frame
        img = frame.copy()  # never draw on the frame itself; later CV stages will use it
        if self.detect_on:
            self.detector.draw(img, self.dim_outside)
        if self.show_hud:
            self.draw_hud(img)
        return img

    def draw_hud(self, img: np.ndarray) -> None:
        det = self.detector
        if not self.detect_on:
            area = "detection off"
        elif det.corners is None:
            area = "searching for projected area..."
        else:
            area = "area LOCKED" if det.locked else "area detected (l to lock)"
        lines = [
            area,
            f"camera {self.cam.width}x{self.cam.height} @ {self.cam.capture_fps:5.1f} fps   "
            f"render {self.render_fps:5.1f} fps   latency {self.latency_ms:5.1f} ms",
            "d detect  l lock  o dim outside  r reset  f fullscreen  h hud  s snap  q quit",
        ]
        y = img.shape[0] - 12 - 24 * (len(lines) - 1)
        for line in lines:
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24

    def handle_key(self, key: int) -> bool:
        """Returns False to quit."""
        if key in (ord("q"), 27):
            return False
        if key == ord("d"):
            self.detect_on = not self.detect_on
        elif key == ord("l"):
            if self.detector.corners is not None:
                self.detector.locked = not self.detector.locked
        elif key == ord("o"):
            self.dim_outside = not self.dim_outside
        elif key == ord("r"):
            self.detector.reset()
        elif key == ord("f"):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord("h"):
            self.show_hud = not self.show_hud
        elif key == ord("s") and self.frame is not None:
            out = ROOT / "snapshots"
            out.mkdir(exist_ok=True)
            path = out / time.strftime("snap_%Y%m%d_%H%M%S.png")
            cv2.imwrite(str(path), self.frame)
            print(f"Saved {path}")
        return True

    def run(self) -> None:
        last_seq = 0
        last_t = time.perf_counter()
        try:
            while self.cam.running:
                f = self.cam.wait_newer(last_seq, timeout=0.03)
                if f is not None:
                    last_seq = f.seq
                    self.frame = f.image
                    cv2.imshow(WINDOW, self.render(f.image))

                    now = time.perf_counter()
                    # Time from the frame leaving the driver to being handed to the window.
                    lat = (now - f.t_capture) * 1000
                    self.latency_ms = 0.9 * self.latency_ms + 0.1 * lat if self.latency_ms else lat
                    dt = now - last_t
                    last_t = now
                    if dt > 0:
                        self.render_fps = 0.9 * self.render_fps + 0.1 / dt if self.render_fps else 1 / dt

                key = cv2.waitKey(1) & 0xFF
                if key != 255 and not self.handle_key(key):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if self.cam.error:
                print(self.cam.error, file=sys.stderr)
        finally:
            self.cam.stop()
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        cams = list_cameras()
        if not cams:
            print("No cameras found.")
        for idx, w, h in cams:
            print(f"camera {idx}: {w}x{h}")
        return
    try:
        App(args).run()
    except RuntimeError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
