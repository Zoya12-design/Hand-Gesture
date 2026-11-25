
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/landmarks.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

FEATURES = [f"f{i}" for i in range(2, 42)]

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("GESTUREX - MODEL TRAINING")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

# Remove training samples with missing subject IDs.
train_df = df[
    (df["split"] == "train") &
    (df["user_id"].notna())
].copy()

val_df = df[df["split"] == "val"].copy()
test_df = df[df["split"] == "test"].copy()

X_train = train_df[FEATURES]
y_train = train_df["label"]

X_val = val_df[FEATURES]
y_val = val_df["label"]

X_test = test_df[FEATURES]
y_test = test_df["label"]

print("\nDATASET SUMMARY")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))
print("Features:", len(FEATURES))
print("Classes:", sorted(y_train.unique()))

# Convert to float32 to reduce memory consumption.
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)


# ============================================================
# CLASSIFIERS
# ============================================================

models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "extra_trees": ExtraTreesClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# ============================================================
# TRAIN AND VALIDATE
# ============================================================

results = []

for name, model in models.items():

    print("\n" + "=" * 65)
    print(f"TRAINING: {name.upper()}")
    print("=" * 65)

    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    start = time.perf_counter()

    predictions = model.predict(X_val)

    prediction_time = time.perf_counter() - start

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print(f"Training time: {training_time:.2f} seconds")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro F1: {macro_f1:.4f}")
    print(f"Validation weighted F1: {weighted_f1:.4f}")
    print(f"Prediction time: {prediction_time:.4f} seconds")

    print("\nCLASSIFICATION REPORT")
    report = classification_report(
        y_val,
        predictions,
        zero_division=0,
    )
    print(report)

    # Save trained model.
    model_path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, model_path)

    # Save validation confusion matrix.
    labels = sorted(y_train.unique())

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=labels,
    )

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).to_csv(REPORT_DIR / f"{name}_validation_confusion_matrix.csv")

    # Save results.
    results.append({
        "model": name,
        "training_time_seconds": training_time,
        "validation_prediction_time_seconds": prediction_time,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "model_path": str(model_path),
    })


# ============================================================
# COMPARE MODELS
# ============================================================

results_df = pd.DataFrame(results).sort_values(
    by="macro_f1",
    ascending=False,
)

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False,
)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nBest model by validation macro F1:")
print(results_df.iloc[0]["model"])

print("\nTraining complete.")
