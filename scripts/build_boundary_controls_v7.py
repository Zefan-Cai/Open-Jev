"""Freeze the original 1792-row v7 boundary candidate without model calls."""
import argparse
import json

from jev.boundary_controls_v7 import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    manifest = build(args.output, args.seed)
    print(json.dumps(manifest["oracle_audit"], indent=2))


if __name__ == "__main__":
    main()
