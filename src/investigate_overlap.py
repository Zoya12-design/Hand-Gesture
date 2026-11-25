
import pandas as pd
from pathlib import Path

DATA = Path("data/processed/landmarks.csv")

df = pd.read_csv(DATA)

features = [f"f{i}" for i in range(42)]

print("=" * 65)
print("GESTUREX - OVERLAP INVESTIGATION")
print("=" * 65)

# Find exact feature-vector matches between train and test
train = df[df["split"] == "train"].copy()
test = df[df["split"] == "test"].copy()

matches = train.merge(
    test,
    on=features,
    suffixes=("_train", "_test")
)

print("\n1. TRAIN-TEST MATCHES")
print("Matching pairs:", len(matches))

if not matches.empty:
    print("\nMatching records:")
    print(
        matches[
            [
                "image_id_train",
                "user_id_train",
                "label_train",
                "image_id_test",
                "user_id_test",
                "label_test",
            ]
        ].to_string(index=False)
    )

    print("\nSame-label matches:")
    print(
        (matches["label_train"] == matches["label_test"])
        .value_counts()
    )

    print("\nSame-image matches:")
    print(
        (matches["image_id_train"] == matches["image_id_test"])
        .value_counts()
    )

    print("\nSame-subject matches:")
    print(
        (matches["user_id_train"] == matches["user_id_test"])
        .value_counts()
    )

# Inspect missing subject IDs
print("\n2. MISSING USER IDS")

missing = df[df["user_id"].isna()]

print("Total:", len(missing))
print("\nDistribution by split:")
print(missing["split"].value_counts())

print("\nDistribution by label:")
print(missing["label"].value_counts())

# Inspect exact duplicate feature vectors and labels
print("\n3. DUPLICATE FEATURE VECTORS + LABELS")

duplicate_mask = df.duplicated(
    subset=features + ["label"],
    keep=False
)

duplicates = df[duplicate_mask].sort_values(
    features + ["label"]
)

print("Rows involved:", len(duplicates))

if not duplicates.empty:
    print(
        duplicates[
            ["split", "image_id", "user_id", "label"]
        ].to_string(index=False)
    )

print("\nINVESTIGATION COMPLETE")
