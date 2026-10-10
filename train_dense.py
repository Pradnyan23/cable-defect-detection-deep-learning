"""Training Dense Neural Network model."""
import argparse
import logging
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam

from src.config.config import DataConfig, TrainConfig, DenseConfig, get_device
from src.datasets import create_dataloaders
from src.models import DenseModel
from src.training import Trainer
from src.evaluation import Evaluator
from src.utils import setup_logging, create_directories


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Dense model")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset_binary"),
        help="Path to dataset_binary",
    )

    args = parser.parse_args()

    # Setup
    device = get_device()
    data_config = DataConfig()
    train_config = TrainConfig(batch_size=args.batch_size, learning_rate=args.lr)
    model_config = DenseConfig(dropout_rate=args.dropout)

    logger = setup_logging(log_dir=train_config.logs_dir, log_name="train_dense")

    create_directories([
        train_config.model_save_dir,
        train_config.logs_dir,
        Path("plots"),
        Path("results"),
    ])

    logger.info("=" * 80)
    logger.info("Training Dense Model")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Dropout: {args.dropout}")

    # Create dataloaders
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset_dir,
        image_size=data_config.image_size,
        train_split=data_config.train_split,
        val_split=data_config.val_split,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        seed=data_config.seed,
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    input_size = data_config.image_size[0] * data_config.image_size[1] * 3
    model = DenseModel(
        input_size=input_size,
        hidden_sizes=model_config.hidden_sizes,
        dropout_rate=model_config.dropout_rate,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    trainer = Trainer(model, device, criterion, optimizer, logger)

    # MLflow setup
    mlflow.set_tracking_uri(train_config.mlflow_tracking_uri)
    mlflow.set_experiment("dense_model")

    with mlflow.start_run():
        mlflow.log_params({
            "batch_size": train_config.batch_size,
            "learning_rate": train_config.learning_rate,
            "epochs": args.epochs,
            "dropout": model_config.dropout_rate,
            "weight_decay": train_config.weight_decay,
            "model_type": "DenseModel",
        })

        # Train
        logger.info("Starting training...")
        history, best_model_path = trainer.fit(
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            checkpoint_dir=train_config.model_save_dir,
            early_stopping_patience=train_config.early_stopping_patience,
            early_stopping_delta=train_config.early_stopping_delta,
        )

        # Save final model
        final_model_path = train_config.model_save_dir / "dense_model_final.pth"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

        # Evaluate
        logger.info("Evaluating on test set...")
        model.load_state_dict(torch.load(best_model_path))
        evaluator = Evaluator(model, device, logger)

        metrics, all_probs, all_labels = evaluator.evaluate(test_loader)

        logger.info("\n" + "=" * 80)
        logger.info("Test Results")
        logger.info("=" * 80)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1']:.4f}")

        # Save plots
        plots_dir = Path("plots")
        evaluator.plot_training_history(history, plots_dir)
        all_preds = (all_probs > 0.5).astype(int)
        evaluator.plot_confusion_matrix(all_labels, all_preds, plots_dir)
        evaluator.plot_roc_curve(all_probs, all_labels, plots_dir)

        # Save metrics
        evaluator.save_metrics(metrics, Path("results"))
        evaluator.print_classification_report(all_labels, all_preds)

        # Log to MLflow
        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "val_loss": history["val_loss"][-1],
            "train_loss": history["train_loss"][-1],
        })

        mlflow.pytorch.log_model(model, "dense_model")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
