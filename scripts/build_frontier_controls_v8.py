"""Build original provisional v8 CPU data; independent approval remains pending."""
import argparse
import json

from jev.frontier_controls_v8 import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261003)
    args = parser.parse_args()
    manifest = build(args.output, args.seed)
    print(json.dumps({'status': manifest['status'], 'summary': manifest['summary'],
                      'oracle_review': manifest['oracle_review'], 'isolation_review': manifest['isolation_review']}))


if __name__ == '__main__':
    main()
