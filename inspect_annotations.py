
import zipfile
import json
from pathlib import Path

ARCHIVE = Path("data/raw/hagrid/annotations.zip")


def inspect_archive():
    with zipfile.ZipFile(ARCHIVE, "r") as archive:

        files = archive.namelist()

        print("=" * 60)
        print("HAGRID ANNOTATION ARCHIVE INSPECTION")
        print("=" * 60)

        # Show annotation organization
        json_files = [
            f for f in files
            if f.endswith(".json")
        ]

        print(f"\nTotal JSON files: {len(json_files)}")

        for split in ["train", "val", "test"]:
            split_files = [
                f for f in json_files
                if f.startswith(f"annotations/{split}/")
            ]
            print(f"{split}: {len(split_files)} files")

        # Inspect one representative annotation
        sample_path = "annotations/train/one.json"

        if sample_path not in files:
            print("\nSample file not found:", sample_path)
            print("\nAvailable train files:")
            print([
                f for f in json_files
                if f.startswith("annotations/train/")
            ][:10])
            return

        print("\n" + "=" * 60)
        print("SAMPLE ANNOTATION")
        print("=" * 60)

        raw = archive.read(sample_path)
        data = json.loads(raw)

        print("\nFile:", sample_path)
        print("Top-level type:", type(data).__name__)

        if isinstance(data, dict):
            print("Number of records:", len(data))

            if data:
                first_id = next(iter(data))
                sample = data[first_id]

                print("\nSample image ID:", first_id)
                print("Sample record type:", type(sample).__name__)

                if isinstance(sample, dict):
                    print("\nRecord fields:")
                    for key, value in sample.items():
                        print(
                            f"  {key}: {type(value).__name__}"
                        )

                    print("\nSample record content:")
                    print(json.dumps(sample, indent=2)[:6000])

        elif isinstance(data, list):
            print("Number of records:", len(data))

            if data:
                sample = data[0]
                print("\nSample record:")
                print(json.dumps(sample, indent=2)[:6000])

        print("\n" + "=" * 60)
        print("INSPECTION COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    inspect_archive()
