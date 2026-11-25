"""GestureX floating desktop control panel.

Run from the project root:
    py -3.14 src/dashboard/floating_dashboard.py

The entire interface is drawn in one OpenCV window. No Streamlit server,
browser UI, or separate camera-processing thread is required.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.cursor_controller import CursorController
from src.core.gesture_engine import GestureEngine


SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"
WINDOW_NAME = "GestureX | Floating Control Panel"

WINDOW_WIDTH = 1180
WINDOW_HEIGHT = 720
PREVIEW_WIDTH = 820
PANEL_X = PREVIEW_WIDTH
PANEL_WIDTH = WINDOW_WIDTH - PANEL_X

# Colors are BGR.
BG = (24, 27, 34)
PANEL_BG = (32, 36, 45)
CARD_BG = (43, 48, 59)
WHITE = (238, 241, 246)
MUTED = (157, 166, 180)
GREEN = (75, 190, 125)
RED = (70, 75, 225)
BLUE = (220, 150, 65)
YELLOW = (40, 205, 245)
DARK = (17, 19, 24)


def load_settings() -> dict[str, Any]:
    with SETTINGS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_settings(settings: dict[str, Any]) -> None:
    with SETTINGS_PATH.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=4)
        file.write("\n")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class FloatingDashboard:
    """Single-window OpenCV UI for GestureX."""

    def __init__(
        self,
        engine: GestureEngine,
        controller: CursorController,
        settings: dict[str, Any],
    ) -> None:
        self.engine = engine
        self.controller = controller
        self.settings = settings

        self.running = True
        self.controls_enabled = False
        self.dragging_slider: str | None = None
        self.last_save_time = 0.0
        self.message = "Ready. Controls are disabled."
        self.message_time = time.monotonic()

        self.slider_specs = {
            "confidence": {
                "label": "Confidence threshold",
                "min": 0.50, "max": 0.99,
                "value": float(settings["recognition"]["confidence_threshold"]),
                "format": "{:.2f}",
            },
            "sensitivity": {
                "label": "Cursor margin",
                "min": 0.00, "max": 0.25,
                "value": float(settings["cursor"]["camera_margin"]),
                "format": "{:.2f}",
            },
            "smoothing": {
                "label": "Maximum smoothing",
                "min": 0.10, "max": 1.00,
                "value": float(settings["cursor"]["max_smoothing"]),
                "format": "{:.2f}",
            },
            "scroll_amount": {
                "label": "Scroll amount",
                "min": 1, "max": 12,
                "value": int(settings["scroll"]["amount"]),
                "format": "{:.0f}",
            },
        }

        self.buttons = {
            "enable": (PANEL_X + 22, 132, PANEL_X + 185, 178),
            "disable": (PANEL_X + 198, 132, PANEL_X + PANEL_WIDTH - 22, 178),
            "pause": (PANEL_X + 22, 190, PANEL_X + 185, 236),
            "emergency": (PANEL_X + 198, 190, PANEL_X + PANEL_WIDTH - 22, 236),
            "save": (PANEL_X + 22, 625, PANEL_X + PANEL_WIDTH - 22, 670),
        }

        self.slider_tracks: dict[str, tuple[int, int, int, int]] = {}

    def set_message(self, message: str) -> None:
        self.message = message
        self.message_time = time.monotonic()

    def _apply_slider_values(self) -> None:
        """Apply UI values to the live engine/controller."""
        values = self.slider_specs

        self.settings["recognition"]["confidence_threshold"] = float(
            values["confidence"]["value"]
        )
        self.settings["cursor"]["camera_margin"] = float(
            values["sensitivity"]["value"]
        )
        self.settings["cursor"]["max_smoothing"] = float(
            values["smoothing"]["value"]
        )
        self.settings["scroll"]["amount"] = int(
            round(values["scroll_amount"]["value"])
        )

        # Update live recognition threshold.
        self.engine.confidence_threshold = float(
            values["confidence"]["value"]
        )

        # Update the live cursor mapping/smoothing.
        cursor = self.controller.cursor
        cursor.margin = float(values["sensitivity"]["value"])
        cursor.max_smoothing = float(values["smoothing"]["value"])

        # Ensure the min/max smoothing relationship remains valid.
        cursor.min_smoothing = min(
            cursor.min_smoothing, cursor.max_smoothing
        )

        # Update live scroll amount.
        self.controller.scroll_amount = int(
            round(values["scroll_amount"]["value"])
        )

    def _save_current_settings(self) -> None:
        self._apply_slider_values()
        save_settings(self.settings)
        self.set_message("Settings saved to config/settings.json")

    def _button_action(self, name: str) -> None:
        if name == "enable":
            self.controller.enable()
            self.controls_enabled = True
            self.set_message("Computer controls ENABLED")
        elif name == "disable":
            self.controller.disable()
            self.controls_enabled = False
            self.set_message("Computer controls disabled")
        elif name == "pause":
            if self.controller.enabled:
                self.controller.toggle_pause()
                self.set_message(
                    "Controls paused" if self.controller.paused
                    else "Controls resumed"
                )
        elif name == "emergency":
            self.controller.emergency_stop()
            self.controls_enabled = False
            self.set_message("EMERGENCY STOP — all controls disabled")
        elif name == "save":
            self._save_current_settings()

    @staticmethod
    def _inside(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
        x, y = point
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _slider_at(self, x: int, y: int) -> str | None:
        for name, rect in self.slider_tracks.items():
            x1, y1, x2, y2 = rect
            if x1 - 8 <= x <= x2 + 8 and y1 - 16 <= y <= y2 + 16:
                return name
        return None

    def _update_slider(self, name: str, x: int) -> None:
        spec = self.slider_specs[name]
        x1, _, x2, _ = self.slider_tracks[name]
        ratio = clamp((x - x1) / max(1, x2 - x1), 0.0, 1.0)
        value = spec["min"] + ratio * (spec["max"] - spec["min"])
        if name == "scroll_amount":
            value = round(value)
        spec["value"] = value
        self._apply_slider_values()

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, rect in self.buttons.items():
                if self._inside((x, y), rect):
                    self._button_action(name)
                    return

            slider_name = self._slider_at(x, y)
            if slider_name is not None:
                self.dragging_slider = slider_name
                self._update_slider(slider_name, x)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_slider:
            self._update_slider(self.dragging_slider, x)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_slider is not None:
                self._update_slider(self.dragging_slider, x)
                self.dragging_slider = None
                self.set_message("Setting adjusted. Press SAVE SETTINGS to persist.")

    def _draw_button(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        label: str,
        color: tuple[int, int, int],
        enabled: bool = True,
    ) -> None:
        x1, y1, x2, y2 = rect
        fill = color if enabled else (70, 73, 80)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (75, 82, 94), 1, cv2.LINE_AA)
        font_scale = 0.48 if len(label) > 13 else 0.55
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )[0]
        tx = x1 + (x2 - x1 - text_size[0]) // 2
        ty = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(
            canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, WHITE, 1, cv2.LINE_AA
        )

    def _draw_slider(
        self, canvas: np.ndarray, name: str, y: int
    ) -> None:
        spec = self.slider_specs[name]
        x1 = PANEL_X + 30
        x2 = PANEL_X + PANEL_WIDTH - 30
        track_y = y + 30
        self.slider_tracks[name] = (x1, track_y, x2, track_y)

        ratio = (
            (float(spec["value"]) - spec["min"])
            / (spec["max"] - spec["min"])
        )
        ratio = clamp(ratio, 0.0, 1.0)
        knob_x = int(x1 + ratio * (x2 - x1))

        cv2.putText(
            canvas, spec["label"], (x1, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, WHITE, 1, cv2.LINE_AA
        )
        value_text = spec["format"].format(spec["value"])
        cv2.putText(
            canvas, value_text, (x2 - 48, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, YELLOW, 1, cv2.LINE_AA
        )
        cv2.line(canvas, (x1, track_y), (x2, track_y), (88, 96, 110), 5, cv2.LINE_AA)
        cv2.line(canvas, (x1, track_y), (knob_x, track_y), BLUE, 5, cv2.LINE_AA)
        cv2.circle(canvas, (knob_x, track_y), 8, WHITE, -1, cv2.LINE_AA)

    def render(
        self,
        camera_frame: np.ndarray,
        recognition: dict[str, Any],
        status: dict[str, Any],
        fps: float,
    ) -> np.ndarray:
        canvas = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), BG, dtype=np.uint8)

        # Left side: live camera preview.
        cv2.rectangle(canvas, (0, 0), (PREVIEW_WIDTH, WINDOW_HEIGHT), DARK, -1)
        cv2.putText(
            canvas, "GESTUREX  /  LIVE CAMERA", (24, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA
        )

        preview_h = 590
        preview_w = PREVIEW_WIDTH - 40
        resized = cv2.resize(camera_frame, (preview_w, preview_h))
        canvas[75:75 + preview_h, 20:20 + preview_w] = resized

        # Overlay a hand position marker for scroll zones when landmarks exist.
        landmarks = recognition.get("landmarks")
        if landmarks and len(landmarks) == 21:
            palm_y = (landmarks[0]["y"] + landmarks[9]["y"]) / 2.0
            marker_y = 75 + int(palm_y * preview_h)
            cv2.line(
                canvas, (20, marker_y), (20 + preview_w, marker_y),
                YELLOW, 1, cv2.LINE_AA
            )

        cv2.putText(
            canvas, "Q: QUIT   E: ENABLE   D: DISABLE   P: PAUSE/RESUME",
            (24, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED, 1, cv2.LINE_AA
        )

        # Right side: floating control panel.
        cv2.rectangle(
            canvas, (PANEL_X, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), PANEL_BG, -1
        )
        px = PANEL_X + 22
        cv2.putText(
            canvas, "CONTROL PANEL", (px, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.68, WHITE, 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas, "AI TOUCHLESS DESKTOP CONTROL", (px, 61),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, MUTED, 1, cv2.LINE_AA
        )

        # Status cards.
        card_y1, card_y2 = 76, 119
        card_w = (PANEL_WIDTH - 58) // 2
        cards = [
            (px, card_y1, px + card_w, card_y2),
            (px + card_w + 12, card_y1, WINDOW_WIDTH - 22, card_y2),
        ]
        for rect in cards:
            cv2.rectangle(canvas, rect[:2], rect[2:], CARD_BG, -1, cv2.LINE_AA)

        gesture_label = str(recognition.get("stable_gesture", "—")).upper()
        confidence = float(recognition.get("confidence", 0.0))
        cv2.putText(canvas, "STABLE GESTURE", (cards[0][0] + 10, 93),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, gesture_label[:13], (cards[0][0] + 10, 111),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, GREEN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "CONFIDENCE", (cards[1][0] + 10, 93),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{confidence * 100:.1f}%", (cards[1][0] + 10, 111),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, YELLOW, 1, cv2.LINE_AA)

        control_color = GREEN if status["status"] == "ACTIVE" else RED
        cv2.putText(
            canvas, f"SYSTEM: {status['status']}", (px, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, control_color, 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas, f"FPS: {fps:.1f}", (WINDOW_WIDTH - 110, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA
        )

        self._draw_button(canvas, self.buttons["enable"], "ENABLE", GREEN,
                          enabled=status["status"] != "ACTIVE")
        self._draw_button(canvas, self.buttons["disable"], "DISABLE", RED,
                          enabled=status["status"] != "DISABLED")
        self._draw_button(canvas, self.buttons["pause"], "PAUSE / RESUME", BLUE,
                          enabled=status["status"] != "DISABLED")
        self._draw_button(canvas, self.buttons["emergency"], "EMERGENCY STOP", RED)

        cv2.putText(
            canvas, "LIVE SETTINGS", (px, 277),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA
        )
        self._draw_slider(canvas, "confidence", 290)
        self._draw_slider(canvas, "sensitivity", 350)
        self._draw_slider(canvas, "smoothing", 410)
        self._draw_slider(canvas, "scroll_amount", 470)

        cv2.putText(
            canvas, "CURRENT ACTION", (px, 548),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED, 1, cv2.LINE_AA
        )
        action = str(status.get("action", "—"))
        cv2.putText(
            canvas, action[:43], (px, 570),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA
        )

        message_color = YELLOW if time.monotonic() - self.message_time < 5 else MUTED
        cv2.putText(
            canvas, self.message[:45], (px, 600),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, message_color, 1, cv2.LINE_AA
        )
        self._draw_button(canvas, self.buttons["save"], "SAVE SETTINGS", BLUE)

        return canvas

    def handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord("e"):
            self._button_action("enable")
        elif key == ord("d"):
            self._button_action("disable")
        elif key == ord("p"):
            self._button_action("pause")
        elif key == ord("s"):
            self._button_action("save")
        return True


def main() -> None:
    settings = load_settings()

    engine = GestureEngine(settings_path=SETTINGS_PATH)
    controller = CursorController(settings_path=SETTINGS_PATH)
    dashboard = FloatingDashboard(engine, controller, settings)

    camera_index = int(settings.get("camera", {}).get("index", 0))
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {camera_index}. "
            "Close other camera applications and try again."
        )

    # Do not force a resolution here. The previous standalone controller
    # used the webcam's default mode, which is generally the most compatible.
    # Some camera drivers report isOpened() == True but fail on read() after
    # an unsupported width/height request.
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"Webcam opened: index={camera_index}, "
        f"resolution={actual_width}x{actual_height}"
    )

    # Give the camera driver a moment to deliver its first frames.
    warmup_frame = None
    for _ in range(30):
        warmup_ok, warmup_frame = camera.read()
        if warmup_ok and warmup_frame is not None and warmup_frame.size:
            break
        time.sleep(0.05)
    else:
        camera.release()
        raise RuntimeError(
            f"Webcam index {camera_index} opened but did not deliver frames. "
            "Close apps using the camera (Camera, Teams, Zoom, browser tabs), "
            "then retry. If needed, change camera.index in config/settings.json."
        )

    frame_times: list[float] = []
    consecutive_read_failures = 0

    try:
        engine.start()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        cv2.setMouseCallback(WINDOW_NAME, dashboard._mouse_callback)

        # Best-effort always-on-top behavior on supported OpenCV/Windows builds.
        try:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1
            )
        except cv2.error:
            pass

        print("\nGESTUREX FLOATING CONTROL PANEL")
        print("E = Enable | D = Disable | P = Pause/Resume | S = Save | Q = Quit")
        print("Controls start DISABLED.")
        print("Move the physical mouse to the top-left corner for PyAutoGUI fail-safe.")

        while dashboard.running:
            success, frame = camera.read()

            if not success or frame is None or frame.size == 0:
                consecutive_read_failures += 1
                if consecutive_read_failures < 10:
                    # Camera drivers can occasionally drop a frame. Retry
                    # briefly rather than terminating the whole application.
                    time.sleep(0.03)
                    continue
                raise RuntimeError(
                    "Webcam repeatedly stopped returning frames. "
                    "Check whether another application is using the camera, "
                    "or try another camera.index in config/settings.json."
                )

            consecutive_read_failures = 0
            display_frame, recognition = engine.process_frame(frame)
            status = controller.process(recognition)

            now = time.monotonic()
            frame_times.append(now)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 0.0
            if len(frame_times) >= 2 and frame_times[-1] > frame_times[0]:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

            canvas = dashboard.render(display_frame, recognition, status, fps)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            dashboard.running = dashboard.handle_key(key)

    finally:
        controller.emergency_stop()
        engine.close()
        camera.release()
        cv2.destroyAllWindows()

    print("GestureX floating dashboard closed safely.")


if __name__ == "__main__":
    main()
