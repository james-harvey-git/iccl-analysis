"""Merge independently generated structured-observer SMC seed caches."""

import argparse
from pathlib import Path

from iccl.analysis.structured_observer.cache import merge_seed_caches


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge structured-observer caches that differ only in their disjoint "
            "SMC seed lists."
        )
    )
    parser.add_argument("cache_paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output_path = merge_seed_caches(args.cache_paths, args.output)
    print(f"merged structured-observer cache: {output_path}")


if __name__ == "__main__":
    main()
