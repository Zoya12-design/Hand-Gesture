
import sys
from pathlib import Path

# Add the GestureX project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.core.gesture_engine import GestureEngine


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    engine = GestureEngine()

    print("\nGestureX Live Engine Test")
    print("Press Q to quit.\n")

    try:
        engine.start()

        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to read camera frame.")
                break

            display_frame, result = engine.process_frame(frame)

            gesture = result["gesture"]
            stable = result["stable_gesture"]
            confidence = result["confidence"]

            cv2.putText(
                display_frame,
                f"Gesture: {gesture.upper()}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Stable: {stable.upper()}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display_frame,
                f"Confidence: {confidence * 100:.1f}%",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 255),
                2,
            )

            cv2.imshow("GestureX - Engine Test", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        engine.close()
        camera.release()
        cv2.destroyAllWindows()

    print("Gesture engine test completed.")


if __name__ == "__main__":
    main()
