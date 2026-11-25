
import zipfile
import json
import numpy as np
import pandas as pd

from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ARCHIVE = Path("data/raw/hagrid/annotations.zip")
OUTPUT = Path("data/processed")

CLASSES = [
    "one",
    "fist",
    "palm",
    "peace",
    "like",
    "two_up",
    "no_gesture",
]

SPLITS = ["train", "val", "test"]

RANDOM_STATE = 42

# Two subjects shared between official train and test.
# Exclude them from test to prevent subject leakage.
EXCLUDED_TEST_SUBJECTS = {
    "ffdb60611fd6e01dc4e170e37e38ed5f",
    "227a878270f058549727f4b10c21eed5",
}

OUTPUT.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    f"f{i}" for i in range(42)
]

# --------------------------------------------------
# LANDMARK VALIDATION
# --------------------------------------------------

def validate_landmarks(hand):

    if not isinstance(hand, list) or len(hand) != 21:
        return False

    try:
        points = np.asarray(hand, dtype=np.float64)
    except (TypeError, ValueError):
        return False

    return (
        points.shape == (21, 2)
        and np.isfinite(points).all()
    )


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def normalize_landmarks(hand):

    points = np.asarray(hand, dtype=np.float64)

    # Landmark 0 = wrist
    wrist = points[0].copy()

    # Translate all landmarks relative to wrist
    points = points - wrist

    # Scale using maximum distance from wrist
    distances = np.linalg.norm(points, axis=1)

    scale = distances.max()

    if scale < 1e-8:
        return None

    points = points / scale

    return points.flatten()


# --------------------------------------------------
# DATASET PROCESSING
# --------------------------------------------------

def process_dataset():

    rows = []
    seen = set()

    counters = {
        "records": 0,
        "mismatches": 0,
        "invalid": 0,
        "duplicates": 0,
        "excluded_subjects": 0,
        "processed": 0,
    }

    print("=" * 65)
    print("GESTUREX - DATA PREPROCESSING")
    print("=" * 65)

    with zipfile.ZipFile(ARCHIVE, "r") as archive:

        files = [
            name for name in archive.namelist()
            if name.endswith(".json")
            and any(
                name.startswith(f"annotations/{split}/")
                for split in SPLITS
            )
        ]

        for file_index, filename in enumerate(files, 1):

            split = filename.split("/")[1]

            print(
                f"\rProcessing annotation files: "
                f"{file_index}/{len(files)}",
                end="",
                flush=True
            )

            data = json.loads(archive.read(filename))

            for image_id, record in data.items():

                counters["records"] += 1

                labels = record.get("labels", [])
                landmarks = record.get(
                    "hand_landmarks", []
                )

                user_id = record.get("user_id", "")

                if len(labels) != len(landmarks):
                    counters["mismatches"] += 1
                    continue

                if (
                    split == "test"
                    and user_id in EXCLUDED_TEST_SUBJECTS
                ):
                    counters["excluded_subjects"] += 1
                    continue

                for hand_index, (label, hand) in enumerate(
                    zip(labels, landmarks)
                ):

                    if label not in CLASSES:
                        continue

                    # Prevent duplicate hand records
                    key = (
                        split,
                        image_id,
                        hand_index,
                        label,
                    )

                    if key in seen:
                        counters["duplicates"] += 1
                        continue

                    seen.add(key)

                    if not validate_landmarks(hand):
                        counters["invalid"] += 1
                        continue

                    features = normalize_landmarks(hand)

                    if features is None:
                        counters["invalid"] += 1
                        continue

                    row = {
                        "image_id": image_id,
                        "user_id": user_id,
                        "split": split,
                        "label": label,
                    }

                    row.update(
                        dict(zip(FEATURE_COLUMNS, features))
                    )

                    rows.append(row)
                    counters["processed"] += 1

    print("\n\nProcessing complete!")

    if not rows:
        raise RuntimeError("No valid samples were extracted.")

    df = pd.DataFrame(rows)

    # --------------------------------------------------
    # SAVE FULL CLEAN DATASET
    # --------------------------------------------------

    output_file = OUTPUT / "landmarks.csv"

    df.to_csv(output_file, index=False)

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("PREPROCESSING REPORT")
    print("=" * 65)

    for name, value in counters.items():
        print(f"{name}: {value:,}")

    print("\nCLASS DISTRIBUTION")

    print(
        df.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nSPLIT DISTRIBUTION")

    print(df["split"].value_counts())

    print("\nFEATURE MATRIX:", df[FEATURE_COLUMNS].shape)

    print("\nOUTPUT:", output_file.resolve())

    print("\nPREPROCESSING FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    process_dataset()
