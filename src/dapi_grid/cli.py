from __future__ import annotations

import argparse
import logging

from .comparison import run_comparison
from .config import load_comparison_config, load_config
from .pipeline import run_pipeline


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dapi-grid",
        description="Whole-tissue DAPI nuclear-shape grid clustering",
    )
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run or resume the whole-slide pipeline")
    run.add_argument("config")
    run.add_argument("--force", action="store_true", help="Reprocess completed chunks")
    compare = sub.add_parser(
        "compare", help="Jointly cluster completed results and compare with a control"
    )
    compare.add_argument("config")
    return p


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    if args.command == "run":
        run_pipeline(load_config(args.config), force=args.force)
    else:
        run_comparison(load_comparison_config(args.config))


if __name__ == "__main__":
    main()
