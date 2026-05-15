"""
Build a compact RGBD1K-style dataset: all major categories, up to 10 sequences each,
100–200 random frames per sequence (same folder layout as source).

Usage:
    python split_dataset.py <source_dataset_root> <output_dataset_root> [--seed 42]

Example:
    python split_dataset.py G:\\UniMod1K-main\\SPT\\data\\RGBD1K_train_labelled G:\\UniMod1K-main\\SPT\\data\\RGBD1K_train_subset
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

try:
    from .split_core import run_split
except ImportError:
    from split_core import run_split


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Subsample UniMod1K/RGBD1K layout: all categories, <=10 sequences each, 100–200 frames."
    )
    parser.add_argument("source_root", type=str, help="e.g. .../RGBD1K_train_labelled")
    parser.add_argument("output_root", type=str, help="Sibling folder next to source, e.g. .../RGBD1K_train_subset")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    args = parser.parse_args(argv)

    source = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()

    if not source.is_dir():
        print(f"ERROR: source is not a directory: {source}", file=sys.stderr)
        return 2
    if source == output:
        print("ERROR: output must differ from source", file=sys.stderr)
        return 2

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    rng = random.Random(seed)

    report = run_split(source, output, rng, max_minors_per_major=10, random_seed=seed)

    print(json.dumps({k: v for k, v in report.items() if k != "majors"}, indent=2, ensure_ascii=False))
    print(f"\nWrote dataset under: {output}")
    print(f"list.txt and split_meta.json are in the output root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
