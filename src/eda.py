
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

DATA = Path("data/processed/landmarks.csv")
REPORTS = Path("reports")

REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

FEATURES = [f"f{i}" for i in range(42)]

print("=" * 65)
print("GESTUREX - EXPLORATORY DATA ANALYSIS")
print("=" * 65)

# --------------------------------------------------
# 1. BASIC INFORMATION
# --------------------------------------------------

print("\n1. DATASET INFORMATION")

print("Shape:", df.shape)
print("\nData types:")
print(df.dtypes.value_counts())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nSplit distribution:")
print(df["split"].value_counts())

# --------------------------------------------------
# 2. MISSING VALUES
# --------------------------------------------------

print("\n2. MISSING VALUES")

missing = df.isnull().sum()

print("Total missing values:", missing.sum())

# --------------------------------------------------
# 3. DUPLICATE ANALYSIS
# --------------------------------------------------

print("\n3. DUPLICATE ANALYSIS")

print(
    "Duplicate rows:",
    df.duplicated(
        subset=["split", "image_id", "label"]
    ).sum()
)

print(
    "Duplicate image IDs:",
    df.duplicated(
        subset=["split", "image_id"]
    ).sum()
)

# --------------------------------------------------
# 4. FEATURE STATISTICS
# --------------------------------------------------

print("\n4. FEATURE STATISTICS")

print(df[FEATURES].describe().round(4).to_string())

print("\nFeature values outside [-1, 1]:",
      ((df[FEATURES] < -1) |
       (df[FEATURES] > 1)).sum().sum())

# --------------------------------------------------
# 5. CLASS DISTRIBUTION CHART
# --------------------------------------------------

plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    x="label",
    order=df["label"].value_counts().index
)

plt.title("GestureX - Gesture Class Distribution")
plt.xlabel("Gesture")
plt.ylabel("Number of Samples")
plt.xticks(rotation=35)
plt.tight_layout()

plt.savefig(REPORTS / "class_distribution.png", dpi=150)
plt.close()

# --------------------------------------------------
# 6. SPLIT DISTRIBUTION
# --------------------------------------------------

split_counts = (
    df.groupby(["split", "label"])
    .size()
    .reset_index(name="samples")
)

plt.figure(figsize=(13, 6))

sns.barplot(
    data=split_counts,
    x="label",
    y="samples",
    hue="split"
)

plt.title("Gesture Distribution Across Dataset Splits")
plt.xlabel("Gesture")
plt.ylabel("Samples")
plt.xticks(rotation=35)
plt.tight_layout()

plt.savefig(REPORTS / "split_distribution.png", dpi=150)
plt.close()

# --------------------------------------------------
# 7. FEATURE CORRELATION
# --------------------------------------------------

sample = df.sample(
    n=min(10000, len(df)),
    random_state=42
)

plt.figure(figsize=(14, 11))

sns.heatmap(
    sample[FEATURES].corr(),
    cmap="coolwarm",
    center=0
)

plt.title("Landmark Feature Correlation")
plt.tight_layout()

plt.savefig(REPORTS / "feature_correlation.png", dpi=150)
plt.close()

# --------------------------------------------------
# 8. FEATURE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(12, 6))

for feature in ["f0", "f1", "f2", "f3", "f4", "f5"]:
    sns.kdeplot(
        data=sample,
        x=feature,
        label=feature
    )

plt.title("Distribution of Selected Landmark Features")
plt.legend()
plt.tight_layout()

plt.savefig(REPORTS / "feature_distribution.png", dpi=150)
plt.close()

# --------------------------------------------------
# 9. SUBJECT OVERLAP VERIFICATION
# --------------------------------------------------

print("\n5. SUBJECT OVERLAP VERIFICATION")

subjects = {
    split: set(df.loc[df["split"] == split, "user_id"])
    for split in ["train", "val", "test"]
}

for first, second in [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]:

    overlap = subjects[first] & subjects[second]

    print(f"{first} vs {second}: {len(overlap)}")

# --------------------------------------------------
# 10. SAVE SUMMARY
# --------------------------------------------------

summary = []

summary.append("GESTUREX EDA REPORT")
summary.append("=" * 50)

summary.append(f"Dataset shape: {df.shape}")
summary.append(f"Missing values: {missing.sum()}")

summary.append("\nCLASS DISTRIBUTION")
summary.append(df["label"].value_counts().to_string())

summary.append("\nSPLIT DISTRIBUTION")
summary.append(df["split"].value_counts().to_string())

summary.append("\nFEATURE STATISTICS")
summary.append(df[FEATURES].describe().to_string())

summary.append("\nSUBJECT OVERLAP")

for first, second in [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]:
    overlap = subjects[first] & subjects[second]
    summary.append(f"{first} vs {second}: {len(overlap)}")

with open(
    REPORTS / "eda_findings.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(summary))

print("\n" + "=" * 65)
print("EDA COMPLETED")
print("Reports saved in:", REPORTS.resolve())
print("=" * 65)
