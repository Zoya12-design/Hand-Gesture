
import mediapipe as mp
import cv2
import sklearn
import joblib

print("=" * 60)
print("GESTUREX - LIVE ENVIRONMENT CHECK")
print("=" * 60)

print("MediaPipe version:", mp.__version__)
print("OpenCV version:", cv2.__version__)
print("Scikit-learn version:", sklearn.__version__)

print("\nMediaPipe API availability:")
print("solutions:", hasattr(mp, "solutions"))
print("tasks:", hasattr(mp, "tasks"))

if hasattr(mp, "solutions"):
    print(
        "solutions.hands:",
        hasattr(mp.solutions, "hands")
    )

print("\nModel loading:")

model = joblib.load("models/extra_trees.joblib")

print("Model loaded successfully")
print("Model classes:", model.classes_)
print("Expected feature count:", model.n_features_in_)

print("\nEnvironment check complete.")
