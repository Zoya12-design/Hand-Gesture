
import pandas as pd
from pathlib import Path

DATA = Path("data/processed/landmarks.csv")

df = pd.read_csv(DATA)

features = [f"f{i}" for i in range(42)]

print("=" * 65)
print("GESTUREX - EDA VALIDATION")
print("=" * 65)

# 1. Missing values by column
print("\n1. MISSING VALUES BY COLUMN")

missing = df.isnull().sum()
print(missing[missing > 0].to_string())

# 2. Missing values in numerical features
print("\n2. MISSING NUMERICAL FEATURES")
print(df[features].isnull().sum().sum())

# 3. Constant features
print("\n3. CONSTANT FEATURES")

constant = [
    col for col in features
    if df[col].nunique(dropna=True) <= 1
]

print(constant)

# 4. Exact duplicate feature vectors and labels
print("\n4. DUPLICATE FEATURE VECTORS")

keys = df[features + ["label"]]

duplicate_mask = keys.duplicated(keep=False)

print("Rows with identical feature vectors and labels:",
      duplicate_mask.sum())

# 5. Duplicate feature vectors across splits
print("\n5. CROSS-SPLIT FEATURE OVERLAP")

hashes = pd.util.hash_pandas_object(
    keys,
    index=False
)

hash_df = pd.DataFrame({
    "hash": hashes,
    "split": df["split"].values
})

split_hashes = {
    split: set(hash_df.loc[
        hash_df["split"] == split, "hash"
    ])
    for split in ["train", "val", "test"]
}

for first, second in [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]:
    overlap = split_hashes[first] & split_hashes[second]

    print(f"{first} vs {second}: {len(overlap)}")

# 6. Non-finite feature values
print("\n6. NON-FINITE FEATURES")

print("Non-finite values:",
      (~df[features].apply(
          lambda col: col.map(
              lambda x: pd.notna(x) and abs(x) != float("inf")
          )
      )).sum().sum())

print("\nVALIDATION COMPLETE")
