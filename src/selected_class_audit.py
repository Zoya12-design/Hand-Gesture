
import zipfile
import json
from collections import defaultdict
from pathlib import Path

ARCHIVE = Path("data/raw/hagrid/annotations.zip")

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

stats = defaultdict(lambda: {
    "valid": 0,
    "invalid": 0,
    "subjects": set(),
})

mismatch_examples = []
subject_ids = {split: set() for split in SPLITS}


def valid_hand(hand):
    if not isinstance(hand, list) or len(hand) != 21:
        return False

    for point in hand:
        if not isinstance(point, list) or len(point) != 2:
            return False

        for value in point:
            if not isinstance(value, (int, float)):
                return False

    return True


def main():

    with zipfile.ZipFile(ARCHIVE, "r") as archive:

        files = [
            name for name in archive.namelist()
            if name.endswith(".json")
        ]

        for filename in files:

            split = filename.split("/")[1]

            if split not in SPLITS:
                continue

            data = json.loads(archive.read(filename))

            for image_id, record in data.items():

                labels = record.get("labels", [])
                landmarks = record.get("hand_landmarks", [])
                user_id = record.get("user_id")

                if user_id:
                    subject_ids[split].add(user_id)

                if len(labels) != len(landmarks):

                    if len(mismatch_examples) < 10:
                        mismatch_examples.append({
                            "image_id": image_id,
                            "split": split,
                            "labels": len(labels),
                            "landmarks": len(landmarks),
                        })

                    continue

                for label, hand in zip(labels, landmarks):

                    if label not in CLASSES:
                        continue

                    item = stats[(split, label)]

                    if user_id:
                        item["subjects"].add(user_id)

                    if valid_hand(hand):
                        item["valid"] += 1
                    else:
                        item["invalid"] += 1

    print("=" * 65)
    print("GESTUREX - SELECTED CLASS AUDIT")
    print("=" * 65)

    print(
        f"\n{'Class':<16} {'Split':<10} "
        f"{'Valid':>10} {'Invalid':>10}"
    )

    for label in CLASSES:
        for split in SPLITS:

            item = stats[(split, label)]

            print(
                f"{label:<16} {split:<10} "
                f"{item['valid']:>10,} "
                f"{item['invalid']:>10,}"
            )

    print("\nSUBJECT OVERLAP")

    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:

            overlap = subject_ids[first] & subject_ids[second]

            print(
                f"{first} vs {second}: {len(overlap)}"
            )

            if overlap:
                print("Shared IDs:", list(overlap)[:5])

    print("\nFIRST 10 LABEL/LANDMARK MISMATCHES")

    for item in mismatch_examples:
        print(item)

    print("\nAUDIT COMPLETE")


if __name__ == "__main__":
    main()
