"""
GestureX computer-control engine.

Consumes recognition results from GestureEngine and performs safe,
configurable mouse actions. Controls are disabled until explicitly enabled.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyautogui


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"


def load_settings(settings_path: str | Path = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    path = Path(settings_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"GestureX settings file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


class AdaptiveCursor:
    """Maps normalized hand coordinates to the screen with adaptive smoothing."""

    def __init__(self, settings: dict[str, Any]) -> None:
        cursor_settings = settings["cursor"]
        self.margin = float(cursor_settings.get("camera_margin", 0.12))
        self.min_smoothing = float(cursor_settings.get("min_smoothing", 0.12))
        self.max_smoothing = float(cursor_settings.get("max_smoothing", 0.65))
        self.speed_reference = float(cursor_settings.get("speed_reference", 180.0))
        self.dead_zone = float(cursor_settings.get("dead_zone", 2.0))

        if not 0 <= self.margin < 0.5:
            raise ValueError("cursor.camera_margin must be between 0 and 0.5.")
        if not 0 < self.min_smoothing <= self.max_smoothing <= 1:
            raise ValueError(
                "Cursor smoothing values must satisfy 0 < min <= max <= 1."
            )
        if self.speed_reference <= 0:
            raise ValueError("cursor.speed_reference must be positive.")

        self.screen_width, self.screen_height = pyautogui.size()
        self.x: float | None = None
        self.y: float | None = None

    def reset(self) -> None:
        self.x = None
        self.y = None

    def move(self, landmark_x: float, landmark_y: float) -> None:
        usable_range = 1.0 - (2.0 * self.margin)

        normalized_x = np.clip(
            (float(landmark_x) - self.margin) / usable_range, 0.0, 1.0
        )
        normalized_y = np.clip(
            (float(landmark_y) - self.margin) / usable_range, 0.0, 1.0
        )

        target_x = float(normalized_x * (self.screen_width - 1))
        target_y = float(normalized_y * (self.screen_height - 1))

        if self.x is None or self.y is None:
            self.x, self.y = target_x, target_y

        dx = target_x - self.x
        dy = target_y - self.y
        distance = float(np.hypot(dx, dy))

        if distance < self.dead_zone:
            return

        speed_factor = min(distance / self.speed_reference, 1.0)
        alpha = self.min_smoothing + (
            (self.max_smoothing - self.min_smoothing) * speed_factor
        )

        self.x += alpha * dx
        self.y += alpha * dy

        pyautogui.moveTo(
            int(np.clip(self.x, 0, self.screen_width - 1)),
            int(np.clip(self.y, 0, self.screen_height - 1)),
            _pause=False,
        )


class CursorController:
    """Executes GestureX actions from GestureEngine result dictionaries."""

    ONE_SHOT_GESTURES = {"like", "peace", "palm"}
    INVALID_GESTURES = {
        "NO HAND",
        "LOW CONFIDENCE",
        "UNCERTAIN",
        "INVALID LANDMARKS",
    }

    def __init__(
        self,
        settings_path: str | Path = DEFAULT_SETTINGS_PATH,
    ) -> None:
        self.settings = load_settings(settings_path)

        safety = self.settings.get("safety", {})
        pyautogui.FAILSAFE = bool(safety.get("pyautogui_failsafe", True))
        pyautogui.PAUSE = 0

        self.cursor = AdaptiveCursor(self.settings)

        clicks = self.settings.get("clicks", {})
        pause_resume = self.settings.get("pause_resume", {})
        scroll = self.settings["scroll"]

        self.click_cooldown = float(clicks.get("cooldown_seconds", 0.75))
        self.palm_cooldown = float(
            pause_resume.get("cooldown_seconds", 1.0)
        )

        self.scroll_upper_limit = float(scroll.get("upper_limit", 0.44))
        self.scroll_lower_limit = float(scroll.get("lower_limit", 0.56))
        self.scroll_interval = float(scroll.get("interval_seconds", 0.055))
        self.scroll_amount = int(scroll.get("amount", 4))

        if not 0 <= self.scroll_upper_limit < self.scroll_lower_limit <= 1:
            raise ValueError(
                "Scroll limits must satisfy 0 <= upper_limit < lower_limit <= 1."
            )
        if self.scroll_interval <= 0 or self.scroll_amount < 1:
            raise ValueError("Scroll interval and amount must be positive.")

        # Never allow computer control to start active.
        self.enabled = False
        self.paused = False

        self.last_action_gesture: str | None = None
        self.last_click_time = 0.0
        self.last_palm_time = 0.0
        self.last_scroll_time = 0.0

        self.action_message = "Controls disabled"

    def enable(self) -> None:
        """Explicitly enable computer control."""
        self.enabled = True
        self.paused = False
        self.last_action_gesture = None
        self.cursor.reset()
        self.action_message = "Controls enabled"

    def disable(self) -> None:
        """Immediately disable all computer-control actions."""
        self.enabled = False
        self.paused = False
        self.last_action_gesture = None
        self.cursor.reset()
        self.action_message = "Controls disabled"

    def emergency_stop(self) -> None:
        """Alias for disable, intended for dashboard emergency-stop action."""
        self.disable()
        self.action_message = "EMERGENCY STOP - controls disabled"

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.action_message = (
            "Controls paused" if self.paused else "Controls resumed"
        )

    def process(self, recognition: dict[str, Any]) -> dict[str, Any]:
        """Process one GestureEngine result and return current control status.

        Expected recognition keys: gesture, stable_gesture, confidence,
        hand_detected, landmarks.
        """
        now = time.monotonic()
        stable_gesture = str(recognition.get("stable_gesture", "NO HAND"))
        landmarks = recognition.get("landmarks")

        if (
            stable_gesture not in self.ONE_SHOT_GESTURES
            and stable_gesture != "UNCERTAIN"
        ):
            self.last_action_gesture = None

        if not self.enabled:
            self.action_message = "Controls disabled"
            return self.get_status(stable_gesture)

        if stable_gesture in self.INVALID_GESTURES:
            if stable_gesture == "NO HAND":
                self.action_message = "No hand detected"
            elif stable_gesture == "LOW CONFIDENCE":
                self.action_message = "Low confidence - no action"
            else:
                self.action_message = "Waiting for stable gesture"
            return self.get_status(stable_gesture)

        # Palm is handled even while paused so it can resume the system.
        if (
            stable_gesture == "palm"
            and self.last_action_gesture != "palm"
            and now - self.last_palm_time >= self.palm_cooldown
        ):
            self.toggle_pause()
            self.last_palm_time = now
            self.last_action_gesture = "palm"
            return self.get_status(stable_gesture)

        if self.paused:
            self.action_message = "Controls paused"
            return self.get_status(stable_gesture)

        if not isinstance(landmarks, list) or len(landmarks) != 21:
            self.action_message = "Landmarks unavailable - no action"
            return self.get_status(stable_gesture)

        if stable_gesture == "one":
            index_tip = landmarks[8]
            self.cursor.move(index_tip["x"], index_tip["y"])
            self.action_message = "Moving cursor"

        elif stable_gesture == "fist":
            self.action_message = "Cursor stopped"

        elif (
            stable_gesture == "like"
            and self.last_action_gesture != "like"
            and now - self.last_click_time >= self.click_cooldown
        ):
            pyautogui.click(button="left", _pause=False)
            self.last_click_time = now
            self.last_action_gesture = "like"
            self.action_message = "Left click"

        elif (
            stable_gesture == "peace"
            and self.last_action_gesture != "peace"
            and now - self.last_click_time >= self.click_cooldown
        ):
            pyautogui.click(button="right", _pause=False)
            self.last_click_time = now
            self.last_action_gesture = "peace"
            self.action_message = "Right click"

        elif stable_gesture == "two_up":
            # Match the stable controller: use the midpoint of wrist (0)
            # and middle-finger MCP (9) for a more representative palm point.
            palm_center_y = (
                float(landmarks[0]["y"]) + float(landmarks[9]["y"])
            ) / 2.0

            if palm_center_y < self.scroll_upper_limit:
                if now - self.last_scroll_time >= self.scroll_interval:
                    pyautogui.scroll(self.scroll_amount, _pause=False)
                    self.last_scroll_time = now
                self.action_message = f"Scrolling UP | palm Y: {palm_center_y:.2f}"

            elif palm_center_y > self.scroll_lower_limit:
                if now - self.last_scroll_time >= self.scroll_interval:
                    pyautogui.scroll(-self.scroll_amount, _pause=False)
                    self.last_scroll_time = now
                self.action_message = (
                    f"Scrolling DOWN | palm Y: {palm_center_y:.2f}"
                )

            else:
                self.action_message = (
                    f"Scroll neutral | palm Y: {palm_center_y:.2f}"
                )

        else:
            self.action_message = f"Recognized: {stable_gesture}"

        return self.get_status(stable_gesture)

    def get_status(self, stable_gesture: str | None = None) -> dict[str, Any]:
        """Return a UI-friendly snapshot of the controller state."""
        return {
            "enabled": self.enabled,
            "paused": self.paused,
            "status": (
                "DISABLED"
                if not self.enabled
                else "PAUSED" if self.paused else "ACTIVE"
            ),
            "stable_gesture": stable_gesture,
            "action": self.action_message,
        }
