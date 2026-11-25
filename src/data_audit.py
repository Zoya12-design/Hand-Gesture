
import zipfile
import json
from collections import Counter, defaultdict
from pathlib import Path

ARCHIVE = Path("data/raw/hagrid/annotations.zip")

splits = ["train", "val", "test"]

class_counts = {split: Counter() for split in splits}
subject_sets = {split: set() for split in splits}

total_images = Counter()
total_hands = Counter()
valid_landmarks = Counter()
invalid_landmarks = Counter()

landmark_shapes = Counter()
label_mismatches = 0

def inspect_dataset():
    global label_mismatches

    print("=" * 65)
    print("GESTUREX - HAGRID DATASET QUALITY AUDIT")
    print("=" * 65)

    with zipfile.ZipFile(ARCHIVE, "r") as archive:

        json_files = [
            name for name in archive.namelist()
            if name.endswith(".json")
            and any(
                name.startswith(f"annotations/{split}/")
                for split in splits
            )
        ]

        for filename in json_files:

            split = filename.split("/")[1]

            data = json.loads(archive.read(filename))

            total_images[split] += len(data)

            for image_id, record in data.items():

                user_id = record.get("user_id")

                if user_id:
                    subject_sets[split].add(user_id)

                labels = record.get("labels", [])
                landmarks = record.get("hand_landmarks", [])

                if len(labels) != len(landmarks):
                    label_mismatches += 1

                for label in labels:
                    class_counts[split][label] += 1
                    total_hands[split] += 1

                for hand in landmarks:

                    if (
                        isinstance(hand, list)
                        and len(hand) == 21
                        and all(
                            isinstance(point, list)
                            and len(point) == 2
                            and all(
                                isinstance(value, (int, float))
                                for value in point
                            )
                            for point in hand
                        )
                    ):
                        valid_landmarks[split] += 1
                        landmark_shapes["21x2"] += 1
                    else:
                        invalid_landmarks[split] += 1

    print("\n1. IMAGE COUNTS")

    for split in splits:
        print(f"{split}: {total_images[split]:,}")

    print("\n2. GESTURE DISTRIBUTION")

    all_classes = sorted({
        label
        for split in splits
        for label in class_counts[split]
    })

    print(f"\n{'Gesture':<22} {'Train':>10} {'Val':>10} {'Test':>10}")

    for label in all_classes:
        print(
            f"{label:<22}"
            f"{class_counts['train'][label]:>10,}"
            f"{class_counts['val'][label]:>10,}"
            f"{class_counts['test'][label]:>10,}"
        )

    print("\n3. LANDMARK QUALITY")

    for split in splits:
        print(
            f"{split}: "
            f"Valid={valid_landmarks[split]:,}, "
            f"Invalid={invalid_landmarks[split]:,}"
        )

    print("\nLandmark shapes:", dict(landmark_shapes))

    print("\n4. UNIQUE SUBJECTS")

    for split in splits:
        print(
            f"{split}: {len(subject_sets[split]):,} unique subjects"
        )

    print("\n5. SUBJECT OVERLAP")

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):

            first = splits[i]
            second = splits[j]

            overlap = subject_sets[first] & subject_sets[second]

            print(
                f"{first} vs {second}: "
                f"{len(overlap):,} shared subjects"
            )

    print("\n6. STRUCTURAL ISSUES")
    print("Label/landmark count mismatches:", label_mismatches)

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    inspect_dataset()
