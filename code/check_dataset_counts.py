"""Print image counts for train/validation emotion folders."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "personal_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check emotion dataset counts.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in ("train", "validation"):
        split_dir = args.dataset_dir / split
        print(f"\n{split}:")
        if not split_dir.is_dir():
            print(f"  missing: {split_dir}")
            continue

        total = 0
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            count = len(list(class_dir.glob("*.*")))
            total += count
            print(f"  {class_dir.name}: {count}")
        print(f"  total: {total}")


if __name__ == "__main__":
    main()
