
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/landmarks.csv")
MODEL_PATH = Path("models/extra_trees.joblib")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)

FEATURES = [f"f{i}" for i in range(2, 42)]


# ============================================================
# LOAD TEST DATA AND SELECTED MODEL
# ============================================================

print("=" * 65)
print("GESTUREX - FINAL TEST EVALUATION")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

test_df = df[df["split"] == "test"].copy()

X_test = test_df[FEATURES].astype(np.float32)
y_test = test_df["label"]

model = joblib.load(MODEL_PATH)

print("\nTEST DATA")
print("Samples:", len(X_test))
print("Features:", len(FEATURES))
print("Classes:", sorted(y_test.unique()))

# ============================================================
# FINAL PREDICTIONS
# ============================================================

start = time.perf_counter()

predictions = model.predict(X_test)

batch_prediction_time = time.perf_counter() - start

accuracy = accuracy_score(y_test, predictions)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0,
)

print("\nFINAL TEST METRICS")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Batch prediction time: {batch_prediction_time:.4f} seconds")
print(
    "Average batch prediction time per sample:",
    f"{batch_prediction_time / len(X_test) * 1000:.6f} ms",
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    predictions,
    zero_division=0,
)

print("\nCLASSIFICATION REPORT")
print(report)

(REPORT_DIR / "final_classification_report.txt").write_text(
    report,
    encoding="utf-8",
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = list(model.classes_)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels,
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels,
)

cm_df.to_csv(
    REPORT_DIR / "final_confusion_matrix.csv"
)

print("\nCONFUSION MATRIX")
print(cm_df)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
)

importance_df.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False,
)

print("\nTOP 10 IMPORTANT FEATURES")
print(importance_df.head(10).to_string(index=False))

# ============================================================
# SAVE FINAL METRICS
# ============================================================

metrics = {
    "model": "ExtraTreesClassifier",
    "test_samples": int(len(X_test)),
    "features": len(FEATURES),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "batch_prediction_time_seconds": float(batch_prediction_time),
    "average_prediction_time_ms": float(
        batch_prediction_time / len(X_test) * 1000
    ),
}

with open(
    REPORT_DIR / "final_test_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)

print("\nFINAL EVALUATION COMPLETE")
print("Reports saved in:", REPORT_DIR.resolve())
