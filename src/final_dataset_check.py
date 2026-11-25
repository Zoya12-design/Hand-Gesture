
import pandas as pd

DATA = "data/processed/landmarks.csv"

df = pd.read_csv(DATA)

features = [f"f{i}" for i in range(42)]

print("=" * 65)
print("GESTUREX - FINAL DATASET CHECK")
print("=" * 65)

# 1. Count unique feature vectors shared across splits
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

train_vectors = train[features].drop_duplicates()
test_vectors = test[features].drop_duplicates()

overlap = train_vectors.merge(
    test_vectors,
    on=features,
    how="inner"
)

print("\n1. UNIQUE TRAIN-TEST OVERLAPPING VECTORS")
print("Unique overlapping vectors:", len(overlap))

# 2. Count rows affected by overlapping vectors
train_overlap = train.merge(
    overlap,
    on=features,
    how="inner"
)

test_overlap = test.merge(
    overlap,
    on=features,
    how="inner"
)

print("Training rows affected:", len(train_overlap))
print("Testing rows affected:", len(test_overlap))

print("\nOverlapping training labels:")
print(train_overlap["label"].value_counts())

print("\nOverlapping testing labels:")
print(test_overlap["label"].value_counts())

# 3. Investigate missing subject IDs
print("\n2. MISSING SUBJECT IDS")
missing = df[df["user_id"].isna()]

print("Rows:", len(missing))
print("Splits:")
print(missing["split"].value_counts())
print("Labels:")
print(missing["label"].value_counts())

# 4. Simulate strict training cleanup
clean_train = train[train["user_id"].notna()].copy()

print("\n3. TRAINING DATA AFTER SUBJECT-ID CLEANUP")
print("Before:", len(train))
print("After:", len(clean_train))
print("Removed:", len(train) - len(clean_train))

print("\nClass distribution:")
print(clean_train["label"].value_counts())

# 5. Final split counts
print("\n4. ORIGINAL SPLIT COUNTS")
print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

print("\nFINAL CHECK COMPLETE")
