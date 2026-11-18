"""Live camera view on the operator screen and the projector.

Run:   python src/main.py                 (auto-detects camera 0 and the projector)
       python src/main.py --list-cameras  (find the Arducam UC-724 index)

Keys (focus either window):
  1  projector shows the raw camera image (letterboxed)
  2  projector shows the camera image warped into projector space (needs calibration)
  3  projector black (avoids the camera->projector feedback "tunnel")
  c  calibrate: click the 4 projected corners in the operator window
  r  reset calibration
  f  toggle projector fullscreen
  h  toggle HUD
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

from calibration import CORNER_LABELS, Calibration, calibration_pattern
from camera import CameraStream, list_cameras
from displays import ProjectorWindow, Screen, detect_screens, letterbox, pick_projector

OPERATOR_WIN = "Operator - camera view"
PROJECTOR_WIN = "Projector"
MODE_RAW, MODE_WARPED, MODE_BLACK = "raw", "warped", "black"
ROOT = Path(__file__).resolve().parent


def parse_size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time camera view for projection mapping")
    p.add_argument("--camera", type=int, default=0, help="camera index (see --list-cameras)")
    p.add_argument("--list-cameras", action="store_true", help="list cameras and exit")
    p.add_argument("--resolution", type=parse_size, default=(1280, 720),
                   help="requested camera resolution WxH (default 1280x720)")
    p.add_argument("--fps", type=int, default=60, help="requested camera FPS")
    p.add_argument("--no-mjpg", action="store_true", help="don't request MJPG from the camera")
    p.add_argument("--projector-size", type=parse_size, default=None,
                   help="projector resolution WxH (default: detected display, else 1280x720)")
    p.add_argument("--projector-pos", type=parse_size, default=None, metavar="XxY",
                   help="top-left of the projector display in desktop coords, e.g. 1440x0")
    p.add_argument("--flip", choices=["none", "h", "v", "hv"], default="none",
                   help="flip projector output (rear / ceiling mounting)")
    p.add_argument("--calibration", type=Path, default=ROOT / "calibration.json")
    return p.parse_args()


class App:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cam = CameraStream(args.camera, *args.resolution, fps=args.fps,
                                mjpg=not args.no_mjpg).start()

        screens = detect_screens()
        projector = pick_projector(screens)
        detected_size = (projector.width, projector.height) if projector else (1280, 720)
        proj_size = args.projector_size or detected_size
        if args.projector_pos is not None:
            projector = Screen(*args.projector_pos, *proj_size, "manual", False)

        for s in screens:
            tag = " <- projector" if projector and (s.x, s.y) == (projector.x, projector.y) else ""
            print(f"display {s.name}: {s.width}x{s.height} at ({s.x},{s.y}){tag}")
        if projector is None:
            print("No second display detected: projector window opens windowed. "
                  "Drag it onto the projector and press 'f'.")

        self.calib = Calibration(args.calibration, proj_size)
        if self.calib.load():
            self.calib.adapt_to_camera((self.cam.width, self.cam.height))
            print(f"Loaded calibration from {args.calibration}")

        cv2.namedWindow(OPERATOR_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(OPERATOR_WIN, min(self.cam.width, 1280), min(self.cam.height, 720))
        cv2.setMouseCallback(OPERATOR_WIN, self.on_mouse)
        self.proj = ProjectorWindow(PROJECTOR_WIN, projector, proj_size)

        self.mode = MODE_WARPED if self.calib.valid else MODE_RAW
        self.calibrating = False
        self.clicks: list[tuple[int, int]] = []
        self.show_hud = True
        self.frame = None
        self.render_fps = 0.0
        self.latency_ms = 0.0
        self.black = np.zeros((proj_size[1], proj_size[0], 3), np.uint8)
        self.pattern = calibration_pattern(*proj_size)

    # ----------------------------------------------------------- calibration
    def on_mouse(self, event, x, y, flags, param) -> None:
        if not self.calibrating or event != cv2.EVENT_LBUTTONDOWN:
            return
        self.clicks.append((x, y))
        if len(self.clicks) == 4:
            self.calib.set_points(self.clicks, (self.cam.width, self.cam.height))
            self.calib.save()
            self.calibrating = False
            self.mode = MODE_WARPED
            print(f"Calibration saved to {self.calib.path}")

    def start_calibration(self) -> None:
        self.calibrating = True
        self.clicks = []

    # -------------------------------------------------------------- rendering
    def render_projector(self, frame: np.ndarray) -> np.ndarray:
        if self.calibrating:
            out = self.pattern
        elif self.mode == MODE_BLACK:
            out = self.black
        elif self.mode == MODE_WARPED and self.calib.valid:
            out = self.calib.warp(frame)
        else:
            out = letterbox(frame, self.calib.proj_w, self.calib.proj_h)
        flip = self.args.flip
        if flip == "h":
            out = cv2.flip(out, 1)
        elif flip == "v":
            out = cv2.flip(out, 0)
        elif flip == "hv":
            out = cv2.flip(out, -1)
        return out

    def render_operator(self, frame: np.ndarray) -> np.ndarray:
        needs_overlay = self.show_hud or self.calibrating or self.calib.valid
        if not needs_overlay:
            return frame
        img = frame.copy()
        if self.calib.valid and not self.calibrating:
            cv2.polylines(img, [self.calib.cam_points.astype(np.int32)], True, (0, 255, 0), 2)
        if self.calibrating:
            for i, pt in enumerate(self.clicks):
                cv2.circle(img, pt, 8, (0, 255, 255), -1)
                cv2.putText(img, str(i + 1), (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            if len(self.clicks) > 1:
                cv2.polylines(img, [np.int32(self.clicks)], False, (0, 255, 255), 2)
            nxt = CORNER_LABELS[len(self.clicks)]
            self.banner(img, f"CALIBRATING - click projected corner {len(self.clicks) + 1} ({nxt})",
                        (0, 200, 255))
        if self.show_hud:
            lines = [
                f"camera {self.cam.width}x{self.cam.height} @ {self.cam.capture_fps:5.1f} fps",
                f"render {self.render_fps:5.1f} fps   pipeline latency {self.latency_ms:5.1f} ms",
                f"projector {self.calib.proj_w}x{self.calib.proj_h}  mode: {self.mode}"
                + ("" if self.calib.valid else " (uncalibrated)"),
                "1 raw  2 warped  3 black  c calibrate  r reset  f fullscreen  h hud  s snap  q quit",
            ]
            y = img.shape[0] - 12 - 26 * (len(lines) - 1)
            for line in lines:
                cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                y += 26
        return img

    @staticmethod
    def banner(img: np.ndarray, text: str, color) -> None:
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ input
    def handle_key(self, key: int) -> bool:
        """Returns False to quit."""
        if key in (ord("q"), 27):
            if self.calibrating:          # Esc cancels calibration first
                self.calibrating = False
                return True
            return False
        if key == ord("1"):
            self.mode = MODE_RAW
        elif key == ord("2"):
            if self.calib.valid:
                self.mode = MODE_WARPED
            else:
                print("No calibration yet - press 'c' first.")
        elif key == ord("3"):
            self.mode = MODE_BLACK
        elif key == ord("c"):
            self.start_calibration()
        elif key == ord("r"):
            self.calib.clear()
            self.calib.path.unlink(missing_ok=True)
            self.mode = MODE_RAW
        elif key == ord("f"):
            self.proj.toggle_fullscreen()
        elif key == ord("h"):
            self.show_hud = not self.show_hud
        elif key == ord("s") and self.frame is not None:
            out = ROOT / "snapshots"
            out.mkdir(exist_ok=True)
            path = out / time.strftime("snap_%Y%m%d_%H%M%S.png")
            cv2.imwrite(str(path), self.frame)
            print(f"Saved {path}")
        return True

    # ------------------------------------------------------------------- loop
    def run(self) -> None:
        last_seq = 0
        last_t = time.perf_counter()
        try:
            while self.cam.running:
                f = self.cam.wait_newer(last_seq, timeout=0.03)
                if f is not None:
                    last_seq = f.seq
                    self.frame = f.image
                    self.proj.show(self.render_projector(f.image))
                    cv2.imshow(OPERATOR_WIN, self.render_operator(f.image))

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
                if cv2.getWindowProperty(OPERATOR_WIN, cv2.WND_PROP_VISIBLE) < 1:
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
