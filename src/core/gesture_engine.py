"""GestureX hand-landmark classification engine.

This module owns MediaPipe hand detection, landmark preprocessing, model
prediction, confidence filtering, and temporal gesture stabilization.

Camera capture and computer-control actions intentionally live elsewhere.
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"
DEFAULT_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "extra_trees.joblib"
DEFAULT_HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"


def load_settings(settings_path: str | Path = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    """Load GestureX settings from JSON."""
    path = Path(settings_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"GestureX settings file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        settings = json.load(file)

    required_sections = {"camera", "recognition", "cursor", "scroll", "safety"}
    missing = required_sections.difference(settings)
    if missing:
        raise ValueError(
            "Settings file is missing required section(s): "
            + ", ".join(sorted(missing))
        )

    return settings


def normalize_landmarks(hand_landmarks: Any) -> np.ndarray | None:
    """Convert 21 MediaPipe landmarks into the model's 40 input features."""
    points = np.asarray(
        [[landmark.x, landmark.y] for landmark in hand_landmarks],
        dtype=np.float32,
    )

    if points.shape != (21, 2):
        return None

    points -= points[0].copy()

    scale = float(np.max(np.linalg.norm(points, axis=1)))
    if scale < 1e-6:
        return None

    points /= scale
    features = points.flatten()[2:]

    if features.shape != (40,) or not np.isfinite(features).all():
        return None

    return features.reshape(1, -1)


class GestureEngine:
    """Detect and classify a hand gesture from each supplied camera frame."""

    def __init__(
        self,
        settings_path: str | Path = DEFAULT_SETTINGS_PATH,
        classifier_path: str | Path = DEFAULT_CLASSIFIER_PATH,
        hand_model_path: str | Path = DEFAULT_HAND_MODEL_PATH,
    ) -> None:
        self.settings = load_settings(settings_path)

        self.classifier_path = self._resolve_project_path(classifier_path)
        self.hand_model_path = self._resolve_project_path(hand_model_path)

        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"Gesture classifier not found: {self.classifier_path}"
            )
        if not self.hand_model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand model not found: {self.hand_model_path}"
            )

        self.classifier = joblib.load(self.classifier_path)

        recognition = self.settings["recognition"]
        self.confidence_threshold = float(
            recognition.get("confidence_threshold", 0.80)
        )
        self.stable_frames = int(recognition.get("stable_frames", 3))
        history_size = int(recognition.get("history_size", 5))

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if self.stable_frames < 1 or history_size < self.stable_frames:
            raise ValueError(
                "history_size must be at least stable_frames, and both must be positive."
            )

        self.gesture_history: deque[str] = deque(maxlen=history_size)
        self.landmarker = None
        self._last_timestamp_ms = -1

    @staticmethod
    def _resolve_project_path(path: str | Path) -> Path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        return resolved

    def start(self) -> "GestureEngine":
        """Initialize the MediaPipe Tasks hand landmarker."""
        if self.landmarker is not None:
            return self

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.hand_model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(options)
        )
        self._last_timestamp_ms = -1
        return self

    def reset_stabilization(self) -> None:
        """Clear temporal history, e.g. when the camera is restarted."""
        self.gesture_history.clear()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int | None = None,
        mirror: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Process one BGR frame.

        Returns the display frame and a result dictionary. The display frame
        is mirrored by default, matching the previous working controller.
        Landmark coordinates correspond to the returned (possibly mirrored)
        frame and are normalized to the [0, 1] image range.
        """
        if self.landmarker is None:
            self.start()

        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("process_frame received an empty camera frame.")

        display_frame = cv2.flip(frame_bgr, 1) if mirror else frame_bgr.copy()
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        if timestamp_ms is None:
            timestamp_ms = time.monotonic_ns() // 1_000_000
        timestamp_ms = int(timestamp_ms)

        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        detection = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        result: dict[str, Any] = {
            "gesture": "NO HAND",
            "stable_gesture": "NO HAND",
            "confidence": 0.0,
            "landmarks": None,
            "hand_detected": False,
        }

        if not detection.hand_landmarks:
            self.reset_stabilization()
            return display_frame, result

        hand = detection.hand_landmarks[0]
        features = normalize_landmarks(hand)

        if features is None:
            self.reset_stabilization()
            result["gesture"] = "INVALID LANDMARKS"
            result["stable_gesture"] = "LOW CONFIDENCE"
            return display_frame, result

        probabilities = self.classifier.predict_proba(features)[0]
        best_index = int(np.argmax(probabilities))
        gesture = str(self.classifier.classes_[best_index])
        confidence = float(probabilities[best_index])

        result.update(
            {
                "gesture": gesture,
                "confidence": confidence,
                "hand_detected": True,
                "landmarks": [
                    {
                        "x": float(point.x),
                        "y": float(point.y),
                        "z": float(point.z),
                    }
                    for point in hand
                ],
            }
        )

        if confidence < self.confidence_threshold:
            self.reset_stabilization()
            result["stable_gesture"] = "LOW CONFIDENCE"
            return display_frame, result

        self.gesture_history.append(gesture)
        counts = Counter(self.gesture_history)
        most_common, count = counts.most_common(1)[0]

        if count >= self.stable_frames:
            result["stable_gesture"] = most_common
        else:
            result["stable_gesture"] = "UNCERTAIN"

        return display_frame, result

    def close(self) -> None:
        """Release the MediaPipe landmarker."""
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
        self.reset_stabilization()

    def __enter__(self) -> "GestureEngine":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
