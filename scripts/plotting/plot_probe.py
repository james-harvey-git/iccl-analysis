"""Render probe comparisons from portable numerical results without model inference."""

import argparse

from iccl.analysis.plotting import plot_probe_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--histories", nargs="*", default=None, help="Training history.json files")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    for path in plot_probe_results(args.results, args.out_dir, histories=args.histories):
        print(path)


if __name__ == "__main__":
    main()
