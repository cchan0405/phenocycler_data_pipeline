from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from .config import load_config
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
    view = sub.add_parser("view", help="Open the local results viewer")
    view.add_argument("results_dir")
    return p


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    if args.command == "run":
        run_pipeline(load_config(args.config), force=args.force)
    else:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            __file__.replace("cli.py", "viewer.py"),
            "--",
            args.results_dir,
        ]
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

