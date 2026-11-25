
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.core.gesture_engine import GestureEngine
from src.core.cursor_controller import CursorController


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    engine = GestureEngine()
    controller = CursorController()

    print("\nGESTUREX INTEGRATION TEST")
    print("E = Enable controls")
    print("D = Disable controls")
    print("Q = Quit")
    print("Controls start DISABLED.")

    try:
        engine.start()

        while True:
            success, frame = camera.read()

            if not success:
                break

            display_frame, recognition = engine.process_frame(frame)

            status = controller.process(recognition)

            cv2.putText(
                display_frame,
                f"Gesture: {recognition['gesture'].upper()}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Confidence: {recognition['confidence'] * 100:.1f}%",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Control: {status['status']}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if status["enabled"] else (0, 0, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Action: {status['action']}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow("GestureX - Integration Test", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("e"):
                controller.enable()
                print("Controls ENABLED.")

            elif key == ord("d"):
                controller.disable()
                print("Controls DISABLED.")

            elif key == ord("q"):
                break

    finally:
        controller.emergency_stop()
        engine.close()
        camera.release()
        cv2.destroyAllWindows()

    print("Integration test completed safely.")


if __name__ == "__main__":
    main()
