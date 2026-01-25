"""Unified CLI (initial scaffold).

Provides basic commands and sets up logging and seeds.
Future work: Wire into training/data/ood modules and script wrappers.
"""

import argparse
from pathlib import Path

from .logging_utils import configure_logger
from .seeds import set_global_seed

# Import existing project utilities
try:
    import detection_utils
except ModuleNotFoundError:
    detection_utils = None  # type: ignore
try:
    import generate_report as gen_report
except ModuleNotFoundError:
    gen_report = None  # type: ignore


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

    # Detect a single image
    detect = sub.add_parser("detect", help="Detect a single image")
    detect.add_argument("image", type=Path, help="Path to image file")

    # Batch detect images in a folder
    batch = sub.add_parser("batch-detect", help="Detect all images in a folder")
    batch.add_argument("folder", type=Path, help="Folder containing images")

    # Generate markdown report from test_images
    sub.add_parser("report", help="Generate markdown report from test_images")

    # Train a model
    train = sub.add_parser("train", help="Train a model (CNN, ART, or FFN)")
    train.add_argument("model_type", choices=["cnn", "art", "ffn"], help="Model type to train")

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

    if args.command == "detect":
        if detection_utils is None:
            logger.error("detection_utils module not available")
            return 1
        logger.info(f"Loading models for detection")
        clf, autoencoder, ood_detector, ae_threshold, model_type = detection_utils.load_models()
        if clf is None:
            return 1
        logger.info(f"Models loaded: {model_type}")
        try:
            pred, conf, belongs, recon, dist, stage = detection_utils.predict_image(
                args.image, clf, autoencoder, ood_detector, ae_threshold
            )
            msg = detection_utils.format_detection_result(
                pred, conf, belongs, recon, dist, stage, ood_detector, ae_threshold, verbose=True
            )
            print(msg)
            return 0
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return 1

    if args.command == "batch-detect":
        if detection_utils is None:
            logger.error("detection_utils module not available")
            return 1
        logger.info(f"Loading models for batch detection")
        clf, autoencoder, ood_detector, ae_threshold, model_type = detection_utils.load_models()
        if clf is None:
            return 1
        logger.info(f"Models loaded: {model_type}")
        # Gather images
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
        files = [p for p in args.folder.iterdir() if p.suffix.lower() in exts]
        if not files:
            logger.warning(f"No images found in {args.folder}")
            return 0
        for p in sorted(files):
            try:
                pred, conf, belongs, recon, dist, stage = detection_utils.predict_image(
                    p, clf, autoencoder, ood_detector, ae_threshold
                )
                compact = detection_utils.format_detection_result(
                    pred, conf, belongs, recon, dist, stage, ood_detector, ae_threshold, verbose=False
                )
                print(f"{p.name}: {compact}")
            except Exception as e:
                logger.error(f"Error processing {p.name}: {e}")
        return 0

    if args.command == "report":
        if gen_report is None:
            logger.error("generate_report module not available")
            return 1
        try:
            gen_report.main()
            return 0
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return 1

    if args.command == "train":
        logger.info(f"Training {args.model_type.upper()} model")
        try:
            if args.model_type == "cnn":
                import nn_train_cnn
                if hasattr(nn_train_cnn, "main"):
                    nn_train_cnn.main()
                else:
                    logger.warning("nn_train_cnn has no main(); executing module")
            elif args.model_type == "art":
                import nn_train_art
                if hasattr(nn_train_art, "main"):
                    nn_train_art.main()
                else:
                    logger.warning("nn_train_art has no main(); executing module")
            elif args.model_type == "ffn":
                import nn_train_ffn
                if hasattr(nn_train_ffn, "main"):
                    nn_train_ffn.main()
                else:
                    logger.warning("nn_train_ffn has no main(); executing module")
            return 0
        except ModuleNotFoundError as e:
            logger.error(f"Training module not found: {e}")
            return 1
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
