import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyautogui


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/extra_trees.joblib")
HAND_MODEL_PATH = Path("models/hand_landmarker.task")

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.80
STABLE_FRAMES = 3
HISTORY_SIZE = 5

CAMERA_MARGIN = 0.12

# Adaptive cursor movement
MIN_SMOOTHING = 0.12
MAX_SMOOTHING = 0.65
SPEED_REFERENCE = 180.0
DEAD_ZONE = 2.0

# Click safety
CLICK_COOLDOWN = 0.75

# Palm pause/resume
PALM_COOLDOWN = 1.0

# Position-based scrolling
# Use the center of the palm (between wrist and middle-finger MCP)
# instead of the wrist alone, which can remain low when the fingers
# are raised into the upper part of the camera view.
SCROLL_UPPER_LIMIT = 0.44
SCROLL_LOWER_LIMIT = 0.56
SCROLL_INTERVAL = 0.055
SCROLL_AMOUNT = 4


# ============================================================
# SAFETY
# ============================================================

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

screen_width, screen_height = pyautogui.size()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading GestureX model...")
classifier = joblib.load(MODEL_PATH)
print("Model loaded successfully.")


# ============================================================
# MEDIAPIPE TASKS
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ============================================================
# LANDMARK PREPROCESSING
# ============================================================

def normalize_landmarks(hand_landmarks):
    points = np.array(
        [[lm.x, lm.y] for lm in hand_landmarks],
        dtype=np.float32,
    )

    if points.shape != (21, 2):
        return None

    wrist = points[0].copy()
    points -= wrist

    distances = np.linalg.norm(points, axis=1)
    scale = np.max(distances)

    if scale < 1e-6:
        return None

    points /= scale
    features = points.flatten()[2:]

    if features.shape != (40,) or not np.isfinite(features).all():
        return None

    return features.reshape(1, -1)


# ============================================================
# ADAPTIVE CURSOR
# ============================================================

class AdaptiveCursor:
    def __init__(self):
        self.x = None
        self.y = None

    def map_position(self, landmark_x, landmark_y):
        usable_range = 1.0 - (2 * CAMERA_MARGIN)

        normalized_x = (landmark_x - CAMERA_MARGIN) / usable_range
        normalized_y = (landmark_y - CAMERA_MARGIN) / usable_range

        normalized_x = np.clip(normalized_x, 0.0, 1.0)
        normalized_y = np.clip(normalized_y, 0.0, 1.0)

        target_x = normalized_x * (screen_width - 1)
        target_y = normalized_y * (screen_height - 1)

        return float(target_x), float(target_y)

    def move(self, landmark_x, landmark_y):
        target_x, target_y = self.map_position(landmark_x, landmark_y)

        if self.x is None:
            self.x = target_x
            self.y = target_y

        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.hypot(dx, dy)

        if distance < DEAD_ZONE:
            return

        speed_factor = min(distance / SPEED_REFERENCE, 1.0)
        alpha = MIN_SMOOTHING + (
            (MAX_SMOOTHING - MIN_SMOOTHING) * speed_factor
        )

        self.x += alpha * dx
        self.y += alpha * dy

        pyautogui.moveTo(
            int(np.clip(self.x, 0, screen_width - 1)),
            int(np.clip(self.y, 0, screen_height - 1)),
            _pause=False,
        )


# ============================================================
# GESTURE STABILIZATION
# ============================================================

gesture_history = deque(maxlen=HISTORY_SIZE)


def get_stable_gesture(gesture):
    gesture_history.append(gesture)
    counts = Counter(gesture_history)
    most_common, count = counts.most_common(1)[0]

    if count >= STABLE_FRAMES:
        return most_common

    return "UNCERTAIN"


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# ============================================================
# CONTROL STATE
# ============================================================

cursor = AdaptiveCursor()

control_enabled = False
paused = False

last_action_gesture = None
last_click_time = 0.0
last_palm_time = 0.0
last_scroll_time = 0.0

action_message = "Waiting for input"

# ============================================================
# MAIN LOOP
# ============================================================

print("\n" + "=" * 60)
print("GESTUREX - TOUCHLESS COMPUTER CONTROL")
print("=" * 60)
print("E = Enable controls")
print("D = Disable controls")
print("Q = Quit")
print("\nControls start DISABLED.")

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            # Mirror the preview. Vertical coordinates are unchanged.
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = time.monotonic_ns() // 1_000_000
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            gesture = "NO HAND"
            stable_gesture = "NO HAND"
            confidence = 0.0
            hand = None
            scroll_position = None
            now = time.monotonic()

            # ----------------------------------------------------
            # HAND DETECTION AND CLASSIFICATION
            # ----------------------------------------------------

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                features = normalize_landmarks(hand)

                if features is not None:
                    probabilities = classifier.predict_proba(features)[0]
                    best_index = int(np.argmax(probabilities))
                    gesture = str(classifier.classes_[best_index])
                    confidence = float(probabilities[best_index])

                    if confidence >= CONFIDENCE_THRESHOLD:
                        stable_gesture = get_stable_gesture(gesture)
                    else:
                        gesture_history.clear()
                        stable_gesture = "LOW CONFIDENCE"
                else:
                    gesture_history.clear()
                    hand = None
            else:
                gesture_history.clear()

            # ----------------------------------------------------
            # RELEASE ONE-SHOT ACTION LATCH
            # ----------------------------------------------------

            action_gestures = {"like", "peace", "palm"}

            if (
                stable_gesture not in action_gestures
                and stable_gesture != "UNCERTAIN"
            ):
                last_action_gesture = None

            # ----------------------------------------------------
            # COMPUTER CONTROL
            # ----------------------------------------------------

            if (
                control_enabled
                and stable_gesture not in {
                    "NO HAND",
                    "LOW CONFIDENCE",
                    "UNCERTAIN",
                }
            ):
                # PALM: PAUSE / RESUME
                if (
                    stable_gesture == "palm"
                    and last_action_gesture != "palm"
                    and now - last_palm_time >= PALM_COOLDOWN
                ):
                    paused = not paused
                    last_palm_time = now
                    last_action_gesture = "palm"
                    action_message = (
                        "Controls paused" if paused else "Controls resumed"
                    )

                # ACTIVE CONTROLS
                elif not paused:
                    if stable_gesture == "one":
                        index_tip = hand[8]
                        cursor.move(index_tip.x, index_tip.y)
                        action_message = "Moving cursor"

                    elif stable_gesture == "fist":
                        action_message = "Cursor stopped"

                    elif (
                        stable_gesture == "like"
                        and last_action_gesture != "like"
                        and now - last_click_time >= CLICK_COOLDOWN
                    ):
                        pyautogui.click(button="left", _pause=False)
                        last_click_time = now
                        last_action_gesture = "like"
                        action_message = "Left click"

                    elif (
                        stable_gesture == "peace"
                        and last_action_gesture != "peace"
                        and now - last_click_time >= CLICK_COOLDOWN
                    ):
                        pyautogui.click(button="right", _pause=False)
                        last_click_time = now
                        last_action_gesture = "peace"
                        action_message = "Right click"

                    elif stable_gesture == "two_up":
                        # Palm center: midpoint of wrist (0) and middle
                        # finger MCP (9). This better represents where
                        # the hand is positioned than wrist-only tracking.
                        palm_center_y = (float(hand[0].y) + float(hand[9].y)) / 2
                        scroll_position = palm_center_y

                        if palm_center_y < SCROLL_UPPER_LIMIT:
                            if now - last_scroll_time >= SCROLL_INTERVAL:
                                # Positive PyAutoGUI scroll = upward.
                                pyautogui.scroll(SCROLL_AMOUNT, _pause=False)
                                last_scroll_time = now
                            action_message = (
                                f"Scrolling UP | palm Y: {palm_center_y:.2f}"
                            )

                        elif palm_center_y > SCROLL_LOWER_LIMIT:
                            if now - last_scroll_time >= SCROLL_INTERVAL:
                                # Negative PyAutoGUI scroll = downward.
                                pyautogui.scroll(-SCROLL_AMOUNT, _pause=False)
                                last_scroll_time = now
                            action_message = (
                                f"Scrolling DOWN | palm Y: {palm_center_y:.2f}"
                            )

                        else:
                            action_message = (
                                f"Scroll neutral | palm Y: {palm_center_y:.2f}"
                            )

                    else:
                        action_message = f"Recognized: {stable_gesture}"

            else:
                if paused:
                    action_message = "Controls paused"
                elif not control_enabled:
                    action_message = "Controls disabled"

            # ----------------------------------------------------
            # DISPLAY
            # ----------------------------------------------------

            height, width, _ = frame.shape

            if not control_enabled:
                control_text = "DISABLED"
            elif paused:
                control_text = "PAUSED"
            else:
                control_text = "ACTIVE"

            cv2.rectangle(frame, (10, 10), (500, 195), (25, 25, 25), -1)

            cv2.putText(
                frame,
                f"Control: {control_text}",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if control_enabled and not paused else (0, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Gesture: {gesture.upper()}",
                (25, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (25, 113),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Action: {action_message}",
                (25, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 220, 255),
                2,
            )

            # Cursor movement boundary
            margin_x = int(width * CAMERA_MARGIN)
            margin_y = int(height * CAMERA_MARGIN)
            cv2.rectangle(
                frame,
                (margin_x, margin_y),
                (width - margin_x, height - margin_y),
                (255, 180, 0),
                2,
            )

            # Scroll zones: green = up, red = down, center = neutral.
            upper_line = int(height * SCROLL_UPPER_LIMIT)
            lower_line = int(height * SCROLL_LOWER_LIMIT)
            cv2.line(frame, (0, upper_line), (width, upper_line), (0, 255, 0), 2)
            cv2.line(frame, (0, lower_line), (width, lower_line), (0, 0, 255), 2)

            cv2.putText(
                frame,
                "UP",
                (width - 65, max(upper_line - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "DOWN",
                (width - 90, min(lower_line + 25, height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            cv2.imshow("GestureX - Touchless Computer Control", frame)

            # ----------------------------------------------------
            # KEYBOARD SAFETY CONTROLS
            # ----------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("e"):
                control_enabled = True
                paused = False
                cursor = AdaptiveCursor()
                gesture_history.clear()
                last_action_gesture = None
                action_message = "Controls enabled"
                print("GestureX controls ENABLED.")

            elif key == ord("d"):
                control_enabled = False
                paused = False
                gesture_history.clear()
                last_action_gesture = None
                action_message = "Controls disabled"
                print("GestureX controls DISABLED.")

            elif key == ord("q"):
                break

finally:
    control_enabled = False
    cap.release()
    cv2.destroyAllWindows()

print("GestureX stopped safely.")
