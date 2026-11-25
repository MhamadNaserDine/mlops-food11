"""
Food-11 data preparation (Lab 1).

Also exposes small *pure* helper functions (no files, no DVC, no GPU)
so they can be unit-tested in CI (Lab 5).
"""
import shutil
from pathlib import Path

from PIL import Image

RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")

SPLITS = ["training", "evaluation", "validation"]

# The list index is the class id used in the raw file names:
# "3_120.jpg" -> class 3 -> "Egg"
CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

IMAGE_SIZE = 128
MINI_LIMIT = 100


# ---------------------------------------------------------------------------
# Pure functions (used by the pipeline below AND by the unit tests)
# ---------------------------------------------------------------------------
def category_from_filename(filename: str) -> str:
    """'3_120.jpg' -> 'Egg'"""
    class_id = int(Path(filename).name.split("_")[0])
    return CLASSES[class_id - 1]  # BUG on purpose (Lab 5, Q9)


def resize_image(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Returns an RGB image of size x size."""
    return img.convert("RGB").resize((size, size))


# ---------------------------------------------------------------------------
# Data preparation (only runs when the file is executed as a script)
# ---------------------------------------------------------------------------
def prepare_output():
    for folder in [PROCESSED_DIR, MINI_DIR]:
        if folder.exists():
            shutil.rmtree(folder)

        for split in SPLITS:
            for class_name in CLASSES:
                (folder / split / class_name).mkdir(parents=True, exist_ok=True)


def process_split(split):
    source_dir = RAW_DIR / split
    mini_counts = {class_name: 0 for class_name in CLASSES}

    files = list(source_dir.iterdir())
    print(f"\nProcessing {split}: {len(files)} images")

    for i, image_path in enumerate(files, start=1):
        if not image_path.is_file():
            continue

        try:
            class_name = category_from_filename(image_path.name)
        except (ValueError, IndexError):
            print(f"Skipping unknown file: {image_path.name}")
            continue

        try:
            with Image.open(image_path) as img:
                img = resize_image(img)

                img.save(PROCESSED_DIR / split / class_name / image_path.name)

                if mini_counts[class_name] < MINI_LIMIT:
                    img.save(MINI_DIR / split / class_name / image_path.name)
                    mini_counts[class_name] += 1

        except (OSError, ValueError) as e:
            print(f"Error processing {image_path}: {e}")

        if i % 500 == 0:
            print(f"Processed {i}/{len(files)}")


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DIR}")

    prepare_output()

    for split in SPLITS:
        process_split(split)

    print("\nDone.")
    print(f"Processed dataset: {PROCESSED_DIR}")
    print(f"Mini dataset: {MINI_DIR}")


if __name__ == "__main__":
    main()