
import urllib.request
from pathlib import Path

URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/"
    "datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
)

DESTINATION = Path("data/raw/hagrid/annotations.zip")
DESTINATION.parent.mkdir(parents=True, exist_ok=True)


def progress(blocks, block_size, total):
    if total <= 0:
        return

    downloaded = min(blocks * block_size, total)
    percent = downloaded / total * 100

    print(
        f"\rDownloaded: {percent:.1f}% | "
        f"{downloaded / 1024**2:.1f} MB",
        end="",
        flush=True
    )


def main():
    print("Downloading HaGRID annotations...")

    try:
        urllib.request.urlretrieve(
            URL,
            DESTINATION,
            reporthook=progress
        )

        print("\n\nDownload complete!")
        print(f"Location: {DESTINATION}")
        print(
            f"Size: {DESTINATION.stat().st_size / 1024**3:.2f} GB"
        )

    except Exception as error:
        print(f"\nDownload failed: {error}")


if __name__ == "__main__":
    main()
