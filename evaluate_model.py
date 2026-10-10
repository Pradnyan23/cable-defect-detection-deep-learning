"""Evaluation script for trained models."""
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.config.config import DataConfig, get_device
from src.datasets import create_dataloaders
from src.models import DenseModel, CNNModel
from src.evaluation import Evaluator
from src.utils import setup_logging, create_directories


def load_dense_model(
    model_path: Path,
    device: torch.device,
) -> DenseModel:
    """Load dense model."""
    input_size = 128 * 128 * 3
    model = DenseModel(
        input_size=input_size,
        hidden_sizes=[512, 256, 64],
        dropout_rate=0.5,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_cnn_model(
    model_path: Path,
    device: torch.device,
) -> CNNModel:
    """Load CNN model."""
    model = CNNModel(
        input_channels=3,
        conv_channels=[32, 64, 128],
        kernel_size=3,
        pool_size=2,
        dropout_rate=0.5,
        input_size=(128, 128),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["dense", "cnn"],
        required=True,
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to saved model",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset_binary"),
        help="Path to dataset_binary",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )

    args = parser.parse_args()

    # Setup
    device = get_device()
    data_config = DataConfig()

    logger = setup_logging(log_dir=Path("logs"), log_name="evaluate_model")

    create_directories([Path("plots"), Path("results")])

    logger.info("=" * 80)
    logger.info(f"Evaluating {args.model.upper()} Model")
    logger.info("=" * 80)

    # Check model path
    if not args.model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        return

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    if args.model == "dense":
        model = load_dense_model(args.model_path, device)
    elif args.model == "cnn":
        model = load_cnn_model(args.model_path, device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloader
    logger.info("Loading dataset...")
    _, _, test_loader = create_dataloaders(
        args.dataset_dir,
        image_size=data_config.image_size,
        train_split=data_config.train_split,
        val_split=data_config.val_split,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        seed=data_config.seed,
    )

    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Evaluate
    logger.info("Evaluating...")
    evaluator = Evaluator(model, device, logger)

    metrics, all_probs, all_labels = evaluator.evaluate(test_loader)

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")

    # Save results
    all_preds = (all_probs > 0.5).astype(int)

    plots_dir = Path("plots")
    evaluator.plot_confusion_matrix(all_labels, all_preds, plots_dir)
    evaluator.plot_roc_curve(all_probs, all_labels, plots_dir)
    evaluator.save_metrics(metrics, Path("results"))
    evaluator.print_classification_report(all_labels, all_preds)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
