"""Binary dataset from raw cable data."""
import argparse
import logging
from pathlib import Path

from src.config.config import DataConfig
from src.data import prepare_binary_dataset
from src.utils import setup_logging


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare binary cable dataset")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data_raw/cable"),
        help="Path to raw dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_binary"),
        help="Path to output directory",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(log_dir=Path("logs"), log_name="prepare_dataset")

    logger.info("Starting dataset preparation...")
    logger.info(f"Raw data directory: {args.raw_data_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    if not args.raw_data_dir.exists():
        logger.error(f"Raw data directory not found: {args.raw_data_dir}")
        return

    # Prepare dataset
    prepare_binary_dataset(args.raw_data_dir, args.output_dir, logger)

    logger.info("Dataset preparation completed!")


if __name__ == "__main__":
    main()
