
import cv2
import mediapipe as mp
import numpy as np
import joblib

from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/extra_trees.joblib")
HAND_MODEL_PATH = Path("models/hand_landmarker.task")

CAMERA_INDEX = 0
MAX_HANDS = 1

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading GestureX model...")

classifier = joblib.load(MODEL_PATH)

print("Classifier loaded.")
print("Classes:", classifier.classes_)

# ============================================================
# MEDIAPIPE TASKS SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ============================================================
# LANDMARK PREPROCESSING
# ============================================================

def normalize_landmarks(hand_landmarks):
    """
    Convert 21 MediaPipe landmarks into 40 model features.

    Uses the same wrist-relative normalization as training:
    - subtract wrist coordinates
    - divide by maximum wrist distance
    - remove wrist coordinates
    """

    points = np.array(
        [[landmark.x, landmark.y] for landmark in hand_landmarks],
        dtype=np.float32
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

    if features.shape != (40,):
        return None

    if not np.isfinite(features).all():
        return None

    return features.reshape(1, -1)


# ============================================================
# CAMERA AND HAND LANDMARKER
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("\nGestureX live recognition started.")
print("Press Q to quit.")

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        frame_timestamp_ms = 0

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            frame_timestamp_ms += 33

            result = landmarker.detect_for_video(
                mp_image,
                frame_timestamp_ms
            )

            gesture_text = "No hand detected"
            confidence_text = ""

            if result.hand_landmarks:

                hand_landmarks = result.hand_landmarks[0]

                features = normalize_landmarks(
                    hand_landmarks
                )

                if features is not None:

                    probabilities = classifier.predict_proba(
                        features
                    )[0]

                    best_index = np.argmax(probabilities)

                    gesture = classifier.classes_[best_index]
                    confidence = probabilities[best_index]

                    gesture_text = gesture.upper()
                    confidence_text = f"{confidence * 100:.1f}%"

                    # Draw hand connections
                    height, width, _ = frame.shape

                    for landmark in hand_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 255, 0),
                            -1
                        )

            # Display prediction
            cv2.rectangle(
                frame,
                (10, 10),
                (420, 105),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                frame,
                f"Gesture: {gesture_text}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence_text}",
                (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow("GestureX - Live Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("GestureX stopped.")
