"""Unified CLI (initial scaffold).

Provides basic commands and sets up logging and seeds.
Future work: Wire into training/data/ood modules and script wrappers.
"""

import argparse
from pathlib import Path

from .logging_utils import configure_logger
from .seeds import set_global_seed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="first_ai", description="Unified CLI for First_AI")
    parser.add_argument("--log-file", type=Path, default=Path("logs/first_ai.log"))
    parser.add_argument("--log-level", type=str, default="info", choices=["debug", "info", "warning", "error"]) 
    parser.add_argument("--seed", type=int, default=42)

    sub = parser.add_subparsers(dest="command", required=True)

    # Simple commands for now
    sub.add_parser("version", help="Show CLI version")
    clean = sub.add_parser("clean", help="Run project cleaner")
    clean.add_argument("--yes", action="store_true", help="Skip prompts and proceed")

    args = parser.parse_args(argv)

    level_map = {"debug": 10, "info": 20, "warning": 30, "error": 40}
    logger = configure_logger("first_ai", level=level_map[args.log_level], to_file=args.log_file)
    set_global_seed(args.seed)
    logger.info("First_AI CLI initialized")

    if args.command == "version":
        print("First_AI CLI scaffold v0.1")
        return 0

    if args.command == "clean":
        # Defer to existing script to avoid duplicates
        try:
            import clean_project as cp
            # If a function exists, use it; else, run module-level logic
            if hasattr(cp, "main"):
                return cp.main(auto_confirm=args.yes)
            else:
                logger.info("Running clean_project script")
                # Fallback: execute script globals
                return 0
        except Exception as e:
            logger.error(f"Cleaner execution failed: {e}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
