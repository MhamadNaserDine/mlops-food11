"""
Data contract for the processed Food-11 dataset (Lab 5).

Usage:
    uv run python src/food11/validate_data.py data/food11_processed_mini
    uv run python src/food11/validate_data.py data/food11_processed --min-per-class 50

Prints one PASS/FAIL line per check and exits with status 1 if any check fails.
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

SPLITS = ["training", "validation", "evaluation"]

# The contract: exactly these 11 class folders in every split
EXPECTED_CLASSES = {
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
}

IMAGE_SIZE = (128, 128)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_SHOWN = 5  # how many problems to print per check


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a processed Food-11 dataset.")
    parser.add_argument("dataset", help="e.g. data/food11_processed_mini")
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def count_images(root: Path) -> dict:
    """counts[split][class] = number of images"""
    return {
        split: {c: len(list_images(root / split / c)) for c in EXPECTED_CLASSES}
        for split in SPLITS
    }


# ---------------------------------------------------------------------------
# The 5 checks: each returns a list of problems (empty list = PASS)
# ---------------------------------------------------------------------------
def check_class_folders(root: Path) -> list[str]:
    problems = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            problems.append(f"{split}: split folder is missing")
            continue
        found = {p.name for p in split_dir.iterdir() if p.is_dir()}
        missing = EXPECTED_CLASSES - found
        extra = found - EXPECTED_CLASSES
        if missing:
            problems.append(f"{split}: missing classes {sorted(missing)}")
        if extra:
            problems.append(f"{split}: unexpected folders {sorted(extra)}")
    return problems


def check_images(root: Path) -> list[str]:
    problems = []
    for split in SPLITS:
        for class_name in sorted(EXPECTED_CLASSES):
            for path in list_images(root / split / class_name):
                try:
                    with Image.open(path) as img:
                        img.load()
                        if img.mode != "RGB" or img.size != IMAGE_SIZE:
                            problems.append(f"{path}: mode={img.mode}, size={img.size}")
                except OSError as e:
                    problems.append(f"{path}: cannot open ({e})")
    return problems


def check_min_per_class(counts: dict, min_per_class: int) -> list[str]:
    return [
        f"{split}/{class_name}: only {n} images"
        for split in SPLITS
        for class_name, n in sorted(counts[split].items())
        if n < min_per_class
    ]


def check_no_duplicates(root: Path) -> list[str]:
    """No image (same file content) may appear in two different splits."""
    seen = defaultdict(set)   # md5 -> splits where it appears
    example = {}              # md5 -> one file path (for the report)
    for split in SPLITS:
        for class_name in sorted(EXPECTED_CLASSES):
            for path in list_images(root / split / class_name):
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                seen[digest].add(split)
                example.setdefault(digest, path)
    return [
        f"{example[d]} also appears in {sorted(splits)}"
        for d, splits in seen.items()
        if len(splits) > 1
    ]


def check_class_balance(counts: dict, tau: float) -> list[str]:
    """Each class share in training must be within tau of its share in validation."""
    train_total = sum(counts["training"].values())
    val_total = sum(counts["validation"].values())
    if train_total == 0 or val_total == 0:
        return ["training or validation split is empty"]
    problems = []
    for class_name in sorted(EXPECTED_CLASSES):
        train_share = counts["training"][class_name] / train_total
        val_share = counts["validation"][class_name] / val_total
        if abs(train_share - val_share) > tau:
            problems.append(
                f"{class_name}: train {train_share:.3f} vs val {val_share:.3f}"
            )
    return problems


def main():
    args = parse_args()
    root = Path(args.dataset)
    if not root.is_dir():
        print(f"Dataset folder not found: {root}")
        sys.exit(1)

    counts = count_images(root)
    total = sum(sum(c.values()) for c in counts.values())
    print(f"Validating {root} ({total} images)\n")

    checks = [
        ("1. each split has exactly the 11 class folders", check_class_folders(root)),
        ("2. every image opens, is RGB and 128x128", check_images(root)),
        (
            f"3. every class has >= {args.min_per_class} images",
            check_min_per_class(counts, args.min_per_class),
        ),
        ("4. no image appears in two splits (md5)", check_no_duplicates(root)),
        (
            f"5. train/val class shares differ by <= {args.tau}",
            check_class_balance(counts, args.tau),
        ),
    ]

    failed = 0
    for name, problems in checks:
        print(f"[{'PASS' if not problems else 'FAIL'}] {name}")
        for p in problems[:MAX_SHOWN]:
            print(f"        - {p}")
        if len(problems) > MAX_SHOWN:
            print(f"        ... and {len(problems) - MAX_SHOWN} more")
        failed += bool(problems)

    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()