# Hyperparameter study for binary classification of cable defects using CNN and Dense NN models.#  Study map:
#      1 → Learning Rate       (CNN) - Yogesh
#      2 → Batch Size          (CNN) - Pradnya
#      3 → Dropout             (CNN) - Sakshee
#      4 → Epochs              (CNN) - Kiran
#      5 → Image Size          (CNN) - Adama
#      6 → Dense vs CNN final comparison - Everyone
#
#  All results are logged to mlflow.db and saved as plots/CSV.
# =============================================================

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION  
# ─────────────────────────────────────────────────────────────
DATASET_DIR: str = "dataset_binary"     # has OK/ and NOT_OK/ subfolders
MLFLOW_URI:  str = "sqlite:///mlflow.db"
RESULTS_DIR: Path = Path("results")
PLOTS_DIR:   Path = Path("plots")
MODELS_DIR:  Path = Path("saved_models")

for _d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    _d.mkdir(exist_ok=True)

SEED:        int   = 42
PATIENCE:    int   = 8        # Early-stopping patience (epochs without gain)
DEFAULT_IMG: int   = 128      # image size for all studies except Person 5
DEFAULT_LR:  float = 1e-3
DEFAULT_BS:  int   = 32
DEFAULT_DO:  float = 0.3
DEFAULT_EP:  int   = 30
DEFAULT_MDL: str   = "cnn"    # model used for studies 1–5


# ─────────────────────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────

def set_seeds(seed: int = SEED) -> None:
    """
    Set all random seeds and enable cuDNN determinism.

    Covers: Python ``random``, NumPy, PyTorch (CPU + GPU).
    Must be called before any model construction or data loading.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Apply at import time so every subsequent call is reproducible.
set_seeds(SEED)


# ─────────────────────────────────────────────────────────────
#  DEVICE  
# ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available compute device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────────────────────────

class DenseNN(nn.Module):
    """
    Compact fully-connected network for binary image classification.

    Kept for the final Dense NN vs CNN comparison (study 6); not used
    in the individual hyperparameter sweeps (studies 1–5).

    Args:
        img_size:     Height / width of square input images.
        hidden_sizes: Sequence of hidden-layer widths.
        dropout:      Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        img_size: int = 128,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        input_dim = img_size * img_size * 3
        layers: List[nn.Module] = [nn.Flatten()]
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x).squeeze(1)


class CNN(nn.Module):
    """
    Three-block convolutional network for binary image classification.

    Architecture:
        Block 1: Conv2d(3→32)  + BN + ReLU + MaxPool
        Block 2: Conv2d(32→64) + BN + ReLU + MaxPool
        Block 3: Conv2d(64→128)+ BN + ReLU + AdaptiveAvgPool(4×4)
        Head   : Linear(2048→256) + ReLU + Dropout → Linear(256→1)

    Args:
        dropout: Dropout probability in the classifier head.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(True), nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.classifier(self.features(x)).squeeze(1)


# ─────────────────────────────────────────────────────────────
#  DATA LOADING  —  stratified 70 / 15 / 15 split
# ─────────────────────────────────────────────────────────────

def get_loaders(
    img_size: int = DEFAULT_IMG,
    batch_size: int = DEFAULT_BS,
    dataset_dir: str = DATASET_DIR,
) -> Tuple[DataLoader, DataLoader, DataLoader, float]:
    """
    Build stratified train / val / test DataLoaders from an ImageFolder.

    Split ratios: 70 % train | 15 % val | 15 % test.
    Class proportions are preserved in every split via
    ``sklearn.model_selection.train_test_split(stratify=targets)``.

    A ``WeightedRandomSampler`` is applied to the training loader so
    each mini-batch is approximately balanced (50 / 50), which prevents
    the model from collapsing to "predict everything as OK".

    Args:
        img_size:    Height / width to resize images to (square).
        batch_size:  Mini-batch size.
        dataset_dir: Root folder containing class sub-directories.

    Returns:
        (train_loader, val_loader, test_loader, pos_weight)
        where *pos_weight* is suitable for ``BCEWithLogitsLoss``.
    """
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Full dataset — only used to obtain class labels for the stratified split.
    full = datasets.ImageFolder(dataset_dir)
    targets: List[int] = full.targets
    all_indices: List[int] = list(range(len(full)))
    print(f"  Classes: {full.class_to_idx}")  # {'NOT_OK': 0, 'OK': 1}

    # ── Stratified 70 / 15 / 15 split ────────────────────────────────────
    # Step 1: train (70 %) vs temp (30 %)
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=SEED,
        stratify=targets,
    )
    # Step 2: split temp equally → val (15 %) and test (15 %)
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_targets,
    )

    # Three separate dataset objects share the same files but carry
    # different transform pipelines (augmentation only for training).
    train_ds = datasets.ImageFolder(dataset_dir, transform=tf_train)
    val_ds   = datasets.ImageFolder(dataset_dir, transform=tf_val)
    test_ds  = datasets.ImageFolder(dataset_dir, transform=tf_val)

    train_subset = Subset(train_ds, train_idx)
    val_subset   = Subset(val_ds,   val_idx)
    test_subset  = Subset(test_ds,  test_idx)

    # WeightedRandomSampler: draw each class with equal probability.
    train_labels: List[int] = [targets[i] for i in train_idx]
    class_counts  = [train_labels.count(0), train_labels.count(1)]
    sample_weights = [1.0 / class_counts[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True,
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, sampler=sampler, num_workers=0,
    )
    val_loader   = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader  = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # pos_weight = (# negatives) / (# positives) for BCEWithLogitsLoss.
    n_pos = class_counts[1]   # OK
    n_neg = class_counts[0]   # NOT_OK
    pos_w = n_neg / n_pos if n_pos > 0 else 1.0

    print(f"  Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")
    print(f"  NOT_OK={class_counts[0]} | OK={class_counts[1]} | pos_weight={pos_w:.3f}")

    return train_loader, val_loader, test_loader, pos_w


# ─────────────────────────────────────────────────────────────
#  TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    training: bool,
) -> Tuple[float, float, float, float, float]:
    """
    Run one forward (and optionally backward) pass over *loader*.

    Args:
        model:     Network to train / evaluate.
        loader:    DataLoader providing (images, labels) batches.
        criterion: Loss function.
        optimizer: Adam optimizer (pass ``None`` during validation).
        device:    Target compute device.
        training:  ``True`` to enable gradients and update weights.

    Returns:
        (avg_loss, accuracy, f1, precision, recall)
    """
    model.train(training)
    total_loss: float = 0.0
    all_preds:  List[int] = []
    all_labels: List[int] = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.float().to(device)
            if training and optimizer is not None:
                optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if training and optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            preds = (torch.sigmoid(logits).detach().cpu() > 0.5).long().tolist()
            all_preds  += preds
            all_labels += labels.cpu().long().tolist()

    avg_loss = total_loss / max(len(loader), 1)
    acc  = float(accuracy_score(all_labels, all_preds))
    f1   = float(f1_score(all_labels, all_preds, zero_division=0))
    prec = float(precision_score(all_labels, all_preds, zero_division=0))
    rec  = float(recall_score(all_labels, all_preds, zero_division=0))
    return avg_loss, acc, f1, prec, rec


def evaluate_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    Evaluate *model* on the held-out test split.

    The model is set to ``eval()`` mode and gradients are disabled.
    Loads the best checkpoint saved during training before calling
    this function.

    Args:
        model:       Network (best checkpoint already loaded).
        test_loader: DataLoader for the held-out test set.
        device:      Target compute device.

    Returns:
        Dictionary with keys:
            accuracy, precision, recall, f1_score, confusion_matrix.
    """
    model.eval()
    all_preds:  List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            preds  = (torch.sigmoid(logits).cpu() > 0.5).long().tolist()
            all_preds  += preds
            all_labels += labels.long().tolist()

    cm = confusion_matrix(all_labels, all_preds)
    return {
        "accuracy":         float(accuracy_score(all_labels, all_preds)),
        "precision":        float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall":           float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1_score":         float(f1_score(all_labels, all_preds, zero_division=0)),
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────────────────────
#  SINGLE TRAINING RUN  (with early stopping on val F1)
# ─────────────────────────────────────────────────────────────

def train_run(
    model_type:   str                   = DEFAULT_MDL,
    lr:           float                 = DEFAULT_LR,
    epochs:       int                   = DEFAULT_EP,
    batch_size:   int                   = DEFAULT_BS,
    dropout:      float                 = DEFAULT_DO,
    img_size:     int                   = DEFAULT_IMG,
    hidden_sizes: Optional[List[int]]   = None,
    experiment:   str                   = "hyperparam",
    run_name:     str                   = "run",
    dataset_dir:  str                   = DATASET_DIR,
    patience:     int                   = PATIENCE,
) -> Tuple[float, Dict, Dict]:
    """
    Train one model configuration with early stopping and test evaluation.

    Early stopping monitors validation F1.  When no improvement is seen
    for *patience* consecutive epochs training halts and the best
    checkpoint is restored automatically before evaluating the test set.

    All per-epoch metrics plus final test metrics are logged to MLflow
    (``sqlite:///mlflow.db``).

    Args:
        model_type:   ``"cnn"`` or ``"dense"``.
        lr:           Learning rate for Adam.
        epochs:       Maximum training epochs (early stopping may halt sooner).
        batch_size:   Mini-batch size.
        dropout:      Dropout probability.
        img_size:     Input image resolution (height == width, square).
        hidden_sizes: Dense NN hidden-layer widths (ignored for CNN).
        experiment:   MLflow experiment name.
        run_name:     MLflow run display name.
        dataset_dir:  Path to the ImageFolder dataset root.
        patience:     Consecutive non-improving epochs before stopping.

    Returns:
        Tuple ``(best_val_f1, history, test_metrics)`` where:
            - *best_val_f1* is the highest val F1 reached during training.
            - *history* contains lists for ``train_loss``, ``val_loss``,
              ``val_acc``, ``val_f1``.
            - *test_metrics* contains ``accuracy``, ``precision``,
              ``recall``, ``f1_score``, and ``confusion_matrix``.
    """
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Run: {run_name}")
    print(f"  model={model_type} lr={lr} bs={batch_size} "
          f"drop={dropout} img={img_size} epochs={epochs} patience={patience}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, pos_w = get_loaders(
        img_size=img_size, batch_size=batch_size, dataset_dir=dataset_dir,
    )

    # ── Build model ──────────────────────────────────────────────────────
    if model_type == "cnn":
        model: nn.Module = CNN(dropout=dropout).to(device)
    else:
        model = DenseNN(
            img_size=img_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w]).to(device),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5,
    )

    # ── MLflow setup ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [],
    }
    best_f1:      float = 0.0
    patience_ctr: int   = 0
    # Safe run_name for filesystem paths
    safe_name  = run_name.replace(" ", "_").replace("/", "-")
    save_path  = MODELS_DIR / f"best_{safe_name}.pt"

    # Save initial weights so save_path always exists before the loop.
    torch.save(model.state_dict(), save_path)

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        params: Dict = {
            "model":      model_type,
            "lr":         lr,
            "epochs":     epochs,
            "batch_size": batch_size,
            "dropout":    dropout,
            "img_size":   img_size,
            "n_params":   n_params,
            "patience":   patience,
        }
        if hidden_sizes:
            params["hidden_sizes"] = str(hidden_sizes)
        mlflow.log_params(params)

        print(f"\n  {'Ep':>3}  {'TrLoss':>8}  {'VLoss':>8}  "
              f"{'VAcc':>6}  {'VF1':>6}  {'Pat':>4}")
        print(f"  {'-'*46}")

        # ── Training loop with early stopping ────────────────────────────
        for epoch in range(epochs):
            tr_loss, tr_acc, tr_f1, _, _ = run_epoch(
                model, train_loader, criterion, optimizer, device, True,
            )
            val_loss, val_acc, val_f1, val_prec, val_rec = run_epoch(
                model, val_loader, criterion, None, device, False,
            )
            scheduler.step(val_f1)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            mlflow.log_metrics(
                {
                    "train_loss":    tr_loss,
                    "val_loss":      val_loss,
                    "val_acc":       val_acc,
                    "val_f1":        val_f1,
                    "val_precision": val_prec,
                    "val_recall":    val_rec,
                },
                step=epoch,
            )

            marker = ""
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_ctr = 0
                torch.save(model.state_dict(), save_path)
                marker = " ✓"
            else:
                patience_ctr += 1

            print(f"  {epoch+1:>3}  {tr_loss:>8.4f}  {val_loss:>8.4f}  "
                  f"{val_acc:>6.4f}  {val_f1:>6.4f}  {patience_ctr:>3}{marker}")

            # ── Early stopping check ──────────────────────────────────────
            if patience_ctr >= patience:
                print(f"\n  Early stopping triggered.")
                break

        # ── Restore best checkpoint, then evaluate on test set ───────────
        model.load_state_dict(torch.load(save_path, map_location=device))
        test_metrics = evaluate_test(model, test_loader, device)

        # Log test-set metrics to MLflow alongside training metrics.
        mlflow.log_metrics({
            "best_val_f1":    best_f1,
            "test_accuracy":  test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall":    test_metrics["recall"],
            "test_f1":        test_metrics["f1_score"],
        })

    print(f"\n  Best val F1 : {best_f1:.4f}")
    print(f"  Test ─ Acc: {test_metrics['accuracy']:.4f}  "
          f"Prec: {test_metrics['precision']:.4f}  "
          f"Rec: {test_metrics['recall']:.4f}  "
          f"F1: {test_metrics['f1_score']:.4f}")
    return best_f1, history, test_metrics


# ─────────────────────────────────────────────────────────────
#  PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────

def plot_bar(
    param_name: str,
    values: List,
    f1_scores: List[float],
    model_type: str,
    filename: str,
) -> None:
    """
    Bar chart showing best validation F1 for each hyperparameter value.

    The bar with the highest F1 is highlighted in a darker colour and a
    dashed reference line is drawn at its height.

    Args:
        param_name: Human-readable parameter name (underscores → spaces).
        values:     Hyperparameter values on the x-axis.
        f1_scores:  Corresponding best validation F1 scores.
        model_type: Model identifier for the chart title.
        filename:   Output file name inside ``PLOTS_DIR``.
    """
    best_f1 = max(f1_scores)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4C72B0" if f == best_f1 else "#8DA0CB" for f in f1_scores]
    bars = ax.bar(
        [str(v) for v in values], f1_scores,
        color=colors, edgecolor="white", linewidth=1.2,
    )
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=11, fontweight="bold")
    ax.set_xlabel(param_name.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Best Validation F1 Score", fontsize=12)
    ax.set_title(
        f"Effect of {param_name.replace('_', ' ')} "
        f"on {model_type.upper()} performance",
        fontsize=13,
    )
    ax.set_ylim(0, 1.08)
    ax.axhline(y=best_f1, color="red", linestyle="--",
               alpha=0.5, linewidth=1, label=f"Best: {best_f1:.3f}")
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out}")


def plot_curves(
    histories: List[Dict],
    labels: List[str],
    param_name: str,
    filename: str,
) -> None:
    """
    Side-by-side loss curves and validation F1 curves for all runs.

    Args:
        histories:  List of history dicts (one per run).
        labels:     Run labels for the legend.
        param_name: Parameter name used in the figure title.
        filename:   Output file name inside ``PLOTS_DIR``.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, (h, label) in enumerate(zip(histories, labels)):
        c = colors[i % len(colors)]
        ax1.plot(h["train_loss"], color=c, linestyle="--",
                 alpha=0.6, label=f"{label} train")
        ax1.plot(h["val_loss"],   color=c, label=f"{label} val")
        ax2.plot(h["val_f1"],     color=c, label=str(label))

    ax1.set_title("Loss curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)

    ax2.set_title(f"Validation F1 — {param_name.replace('_', ' ')}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.legend(fontsize=8)

    plt.suptitle(f"Hyperparameter study: {param_name.replace('_', ' ')}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curves saved → {out}")


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    filename: str,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Plot and save a labelled confusion matrix heatmap.

    Args:
        cm:           2-D confusion matrix (rows = true, cols = predicted).
        title:        Figure title.
        filename:     Output file name inside ``PLOTS_DIR``.
        class_names:  Class labels (default: ['NOT_OK', 'OK']).
    """
    if class_names is None:
        class_names = ["NOT_OK", "OK"]

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    out = PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {out}")


# ─────────────────────────────────────────────────────────────
#  CSV HELPER
# ─────────────────────────────────────────────────────────────

def save_results_csv(
    param_name:  str,
    values:      List,
    all_metrics: List[Dict],
    experiment:  str,
) -> pd.DataFrame:
    """
    Persist study results to ``results/study_{param_name}.csv``.

    Columns: ``{param_name}``, best_val_f1, accuracy, precision,
             recall, f1_score, rank.
    Rows are sorted by *f1_score* (test set) **descending** and a
    numeric rank column (1 = best) is added.

    Args:
        param_name:  Name of the hyperparameter studied.
        values:      Hyperparameter values tested.
        all_metrics: List of metric dicts (one per value); must contain
                     keys: best_val_f1, accuracy, precision, recall,
                     f1_score.
        experiment:  MLflow experiment name (stored for reference).

    Returns:
        The persisted DataFrame.
    """
    df = pd.DataFrame({
        param_name:    values,
        "best_val_f1": [m["best_val_f1"] for m in all_metrics],
        "accuracy":    [m["accuracy"]     for m in all_metrics],
        "precision":   [m["precision"]    for m in all_metrics],
        "recall":      [m["recall"]       for m in all_metrics],
        "f1_score":    [m["f1_score"]     for m in all_metrics],
    })
    # Rank by test F1 (1 = best), ties use min rank.
    df["rank"] = (
        df["f1_score"].rank(ascending=False, method="min").astype(int)
    )
    df = df.sort_values("f1_score", ascending=False).reset_index(drop=True)

    csv_path = RESULTS_DIR / f"study_{param_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved → {csv_path}")
    print(f"\n  {'─'*60}")
    print(df.to_string(index=False))
    print(f"  {'─'*60}")
    best_row = df.iloc[0]
    print(f"  Best {param_name}: {best_row[param_name]}"
          f" (Test F1={best_row['f1_score']:.4f}, "
          f"Val F1={best_row['best_val_f1']:.4f})")
    return df


# ─────────────────────────────────────────────────────────────
#  STUDY FUNCTIONS  —  one per team member
# ─────────────────────────────────────────────────────────────

# ── Yogesh — Learning rate ─────────────────────────────────

def study_learning_rate() -> None:
    """
    Study the effect of learning rate on CNN performance.

    Fixed: batch=32, dropout=0.3, epochs=DEFAULT_EP, img=128, model=CNN.
    Sweep: lr ∈ {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}.

    Insight:
        Too high → unstable training, loss diverges.
        Too low  → slow convergence, may not reach best accuracy.

    Artifacts (plots/):
        study_lr_bar.png, study_lr_curves.png, study_lr_cm.png
    Artifacts (results/):
        study_learning_rate.csv
    """
    print("\n" + "╔" + "═"*56 + "╗")
    print("║  PERSON 1 — Learning Rate Study                       ║")
    print("╚" + "═"*56 + "╝")

    values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    f1s:         List[float] = []
    histories:   List[Dict]  = []
    all_metrics: List[Dict]  = []
    cms:         List[np.ndarray] = []
    exp = "study_learning_rate"

    for lr in values:
        best_f1, h, test_m = train_run(
            model_type="cnn",
            lr=lr,
            run_name=f"lr={lr}",
            experiment=exp,
        )
        f1s.append(best_f1)
        histories.append(h)
        all_metrics.append({
            "best_val_f1": best_f1,
            "accuracy":    test_m["accuracy"],
            "precision":   test_m["precision"],
            "recall":      test_m["recall"],
            "f1_score":    test_m["f1_score"],
        })
        cms.append(test_m["confusion_matrix"])

    plot_bar("learning_rate", values, f1s, "cnn", "study_lr_bar.png")
    plot_curves(histories, [f"lr={v}" for v in values],
                "learning_rate", "study_lr_curves.png")

    # Confusion matrix for the best-performing configuration.
    best_idx = int(np.argmax(f1s))
    plot_confusion_matrix(
        cms[best_idx],
        title=f"LR Study — Best lr={values[best_idx]} "
              f"(Test F1={all_metrics[best_idx]['f1_score']:.3f})",
        filename="study_lr_cm.png",
    )

    save_results_csv("learning_rate", values, all_metrics, exp)


# ── Pradnya — Batch size ────────────────────────────────────

def study_batch_size() -> None:
    """
    Study the effect of batch size on CNN performance.

    Fixed: lr=1e-3, dropout=0.3, epochs=DEFAULT_EP, img=128, model=CNN.
    Sweep: batch_size ∈ {8, 16, 32, 64}.

    Insight:
        Small batches → noisy gradients, often better generalisation.
        Large batches → stable updates, can overfit on small datasets.

    Artifacts (plots/):
        study_bs_bar.png, study_bs_curves.png, study_bs_cm.png
    Artifacts (results/):
        study_batch_size.csv
    """
    print("\n" + "╔" + "═"*56 + "╗")
    print("║  PERSON 2 — Batch Size Study                          ║")
    print("╚" + "═"*56 + "╝")

    values = [8, 16, 32, 64]
    f1s:         List[float] = []
    histories:   List[Dict]  = []
    all_metrics: List[Dict]  = []
    cms:         List[np.ndarray] = []
    exp = "study_batch_size"

    for bs in values:
        best_f1, h, test_m = train_run(
            model_type="cnn",
            batch_size=bs,
            run_name=f"batch={bs}",
            experiment=exp,
        )
        f1s.append(best_f1)
        histories.append(h)
        all_metrics.append({
            "best_val_f1": best_f1,
            "accuracy":    test_m["accuracy"],
            "precision":   test_m["precision"],
            "recall":      test_m["recall"],
            "f1_score":    test_m["f1_score"],
        })
        cms.append(test_m["confusion_matrix"])

    plot_bar("batch_size", values, f1s, "cnn", "study_bs_bar.png")
    plot_curves(histories, [f"bs={v}" for v in values],
                "batch_size", "study_bs_curves.png")

    best_idx = int(np.argmax(f1s))
    plot_confusion_matrix(
        cms[best_idx],
        title=f"Batch Size Study — Best bs={values[best_idx]} "
              f"(Test F1={all_metrics[best_idx]['f1_score']:.3f})",
        filename="study_bs_cm.png",
    )

    save_results_csv("batch_size", values, all_metrics, exp)


# ── Sakshee — Dropout rate ──────────────────────────────────

def study_dropout() -> None:
    """
    Study the effect of dropout regularisation on CNN performance.

    Fixed: lr=1e-3, batch=32, epochs=DEFAULT_EP, img=128, model=CNN.
    Sweep: dropout ∈ {0.0, 0.1, 0.3, 0.5, 0.7}.

    Insight:
        dropout=0.0 → risk of overfitting on small datasets.
        dropout high → underfitting / slow convergence.

    Artifacts (plots/):
        study_dropout_bar.png, study_dropout_curves.png,
        study_dropout_cm.png
    Artifacts (results/):
        study_dropout.csv
    """
    print("\n" + "╔" + "═"*56 + "╗")
    print("║  PERSON 3 — Dropout Study                             ║")
    print("╚" + "═"*56 + "╝")

    values = [0.0, 0.1, 0.3, 0.5, 0.7]
    f1s:         List[float] = []
    histories:   List[Dict]  = []
    all_metrics: List[Dict]  = []
    cms:         List[np.ndarray] = []
    exp = "study_dropout"

    for dr in values:
        best_f1, h, test_m = train_run(
            model_type="cnn",
            dropout=dr,
            run_name=f"dropout={dr}",
            experiment=exp,
        )
        f1s.append(best_f1)
        histories.append(h)
        all_metrics.append({
            "best_val_f1": best_f1,
            "accuracy":    test_m["accuracy"],
            "precision":   test_m["precision"],
            "recall":      test_m["recall"],
            "f1_score":    test_m["f1_score"],
        })
        cms.append(test_m["confusion_matrix"])

    plot_bar("dropout", values, f1s, "cnn", "study_dropout_bar.png")
    plot_curves(histories, [f"do={v}" for v in values],
                "dropout", "study_dropout_curves.png")

    best_idx = int(np.argmax(f1s))
    plot_confusion_matrix(
        cms[best_idx],
        title=f"Dropout Study — Best dropout={values[best_idx]} "
              f"(Test F1={all_metrics[best_idx]['f1_score']:.3f})",
        filename="study_dropout_cm.png",
    )

    save_results_csv("dropout", values, all_metrics, exp)


# ── Kiran — Training epochs  ──

def study_epochs() -> None:
    """
    Study the effect of maximum training epochs on CNN performance.

    Fixed: lr=1e-3, batch=32, dropout=0.3, img=128, model=CNN.
    Sweep: epochs ∈ {20, 40, 60, 80}  (early stopping may halt sooner).

    Because early stopping with ``patience=PATIENCE`` is always active,
    the actual number of epochs trained may be less than the maximum.
    This study reveals whether the model benefits from a larger training
    budget and how quickly it converges.

    Insight:
        Too few epochs → under-trained, validation F1 still rising.
        More epochs + early stopping → optimal checkpoint is rescued
        automatically without manual tuning.

    Artifacts (plots/):
        study_epochs_bar.png, study_epochs_curves.png,
        study_epochs_cm.png
    Artifacts (results/):
        study_epochs.csv
    """
    print("\n" + "╔" + "═"*56 + "╗")
    print("║  PERSON 4 — Epochs Study                              ║")
    print("╚" + "═"*56 + "╝")

    values = [20, 40, 60, 80]
    f1s:         List[float] = []
    histories:   List[Dict]  = []
    all_metrics: List[Dict]  = []
    cms:         List[np.ndarray] = []
    exp = "study_epochs"

    for ep in values:
        best_f1, h, test_m = train_run(
            model_type="cnn",
            lr=1e-3,
            batch_size=32,
            dropout=0.3,
            img_size=128,
            epochs=ep,
            run_name=f"epochs={ep}",
            experiment=exp,
        )
        f1s.append(best_f1)
        histories.append(h)
        all_metrics.append({
            "best_val_f1": best_f1,
            "accuracy":    test_m["accuracy"],
            "precision":   test_m["precision"],
            "recall":      test_m["recall"],
            "f1_score":    test_m["f1_score"],
        })
        cms.append(test_m["confusion_matrix"])

    plot_bar("epochs", values, f1s, "cnn", "study_epochs_bar.png")
    plot_curves(histories, [f"ep={v}" for v in values],
                "epochs", "study_epochs_curves.png")

    best_idx = int(np.argmax(f1s))
    plot_confusion_matrix(
        cms[best_idx],
        title=f"Epochs Study — Best max_epochs={values[best_idx]} "
              f"(Test F1={all_metrics[best_idx]['f1_score']:.3f})",
        filename="study_epochs_cm.png",
    )

    save_results_csv("epochs", values, all_metrics, exp)


# ── Adama— Image size ────────────────────────────────────

def study_image_size() -> None:
    """
    Study the effect of input image resolution on CNN performance.

    Fixed: lr=1e-3, batch=32, dropout=0.3, epochs=DEFAULT_EP, model=CNN.
    Sweep: img_size ∈ {32, 64, 128}.

    Note: 224 px is excluded on CPU — training would take hours.

    Insight:
        Larger images ≠ always better with only ~374 images.
        32 px → very fast, loses fine cable-surface detail.
        64 px → good speed / accuracy balance.
        128 px → best detail, but slower and higher overfitting risk.

    Artifacts (plots/):
        study_imgsize_bar.png, study_imgsize_curves.png,
        study_imgsize_cm.png
    Artifacts (results/):
        study_image_size.csv
    """
    print("\n" + "╔" + "═"*56 + "╗")
    print("║  PERSON 5 — Image Size Study                          ║")
    print("╚" + "═"*56 + "╝")

    values = [32, 64, 128]
    f1s:         List[float] = []
    histories:   List[Dict]  = []
    all_metrics: List[Dict]  = []
    cms:         List[np.ndarray] = []
    exp = "study_image_size"

    for sz in values:
        best_f1, h, test_m = train_run(
            model_type="cnn",
            img_size=sz,
            run_name=f"img={sz}x{sz}",
            experiment=exp,
        )
        f1s.append(best_f1)
        histories.append(h)
        all_metrics.append({
            "best_val_f1": best_f1,
            "accuracy":    test_m["accuracy"],
            "precision":   test_m["precision"],
            "recall":      test_m["recall"],
            "f1_score":    test_m["f1_score"],
        })
        cms.append(test_m["confusion_matrix"])

    plot_bar("image_size", [f"{v}x{v}" for v in values],
             f1s, "cnn", "study_imgsize_bar.png")
    plot_curves(histories, [f"{v}×{v}" for v in values],
                "image_size", "study_imgsize_curves.png")

    best_idx = int(np.argmax(f1s))
    plot_confusion_matrix(
        cms[best_idx],
        title=f"Image Size Study — Best {values[best_idx]}×{values[best_idx]} px "
              f"(Test F1={all_metrics[best_idx]['f1_score']:.3f})",
        filename="study_imgsize_cm.png",
    )

    save_results_csv("image_size", values, all_metrics, exp)


# ─────────────────────────────────────────────────────────────
#  STUDY 6 — FINAL COMPARISON: Dense NN vs CNN - Everyone
# ─────────────────────────────────────────────────────────────

def compare_dense_vs_cnn() -> None:
    """
    Train both architectures with their best hyperparameters and produce
    a side-by-side comparison for the term paper.

    Dense NN best config (from prior studies):
        img_size=64, hidden=[256,128], epochs=40, lr=1e-3,
        batch=32, dropout=0.3.

    CNN best config:
        img_size=128, epochs=40, lr=1e-3, batch=32, dropout=0.3.

    Compares:  Accuracy | Precision | Recall | F1 Score (test set).

    Artifacts (plots/):
        final_comparison.png
    Artifacts (results/):
        final_comparison_metrics.csv
    """
    print("\n" + "╔" + "═"*56 + "╗")
    print("║  STUDY 6 — Dense NN vs CNN final comparison           ║")
    print("╚" + "═"*56 + "╝")

    # ── Train both models ────────────────────────────────────────────────
    f1_dense, h_dense, test_dense = train_run(
        model_type="dense",
        img_size=64,
        hidden_sizes=[256, 128],
        epochs=40,
        run_name="dense_final",
        experiment="model_comparison",
    )
    f1_cnn, h_cnn, test_cnn = train_run(
        model_type="cnn",
        img_size=128,
        epochs=40,
        run_name="cnn_final",
        experiment="model_comparison",
    )

    # ── final_comparison_metrics.csv ────────────────────────────────────
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    comparison_df = pd.DataFrame(
        {
            "model":       ["Dense NN", "CNN"],
            "best_val_f1": [f1_dense, f1_cnn],
            "accuracy":    [test_dense["accuracy"],   test_cnn["accuracy"]],
            "precision":   [test_dense["precision"],  test_cnn["precision"]],
            "recall":      [test_dense["recall"],     test_cnn["recall"]],
            "f1_score":    [test_dense["f1_score"],   test_cnn["f1_score"]],
        }
    )
    csv_path = RESULTS_DIR / "final_comparison_metrics.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n  Final comparison CSV → {csv_path}")
    print(comparison_df.to_string(index=False))

    winner = "CNN" if f1_cnn >= f1_dense else "Dense NN"
    print(f"\n  Winner: {winner}")

    # ── final_comparison.png ─────────────────────────────────────────────
    # Layout: row 1 = loss/F1 curves, row 2 = grouped bar chart of all
    # test-set metrics.
    fig = plt.figure(figsize=(14, 8))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)

    ax_loss  = fig.add_subplot(gs[0, 0])
    ax_f1    = fig.add_subplot(gs[0, 1])
    ax_bar   = fig.add_subplot(gs[1, :])   # full-width bar chart

    colors_model = {"Dense NN": "#4C72B0", "CNN": "#DD8452"}

    for (h, label, test_m) in [
        (h_dense, "Dense NN", test_dense),
        (h_cnn,   "CNN",      test_cnn),
    ]:
        c = colors_model[label]
        ax_loss.plot(h["val_loss"], color=c, label=label)
        ax_f1.plot(h["val_f1"],    color=c, label=label)

    ax_loss.set_title("Validation Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_f1.set_title("Validation F1 Score")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("F1")
    ax_f1.legend()

    # Grouped bar chart of test-set metrics
    x       = np.arange(len(metric_keys))
    bar_w   = 0.35
    vals_dense = [test_dense[k] for k in metric_keys]
    vals_cnn   = [test_cnn[k]   for k in metric_keys]

    bars_d = ax_bar.bar(x - bar_w / 2, vals_dense, bar_w,
                        label="Dense NN", color=colors_model["Dense NN"],
                        edgecolor="white")
    bars_c = ax_bar.bar(x + bar_w / 2, vals_cnn,   bar_w,
                        label="CNN",      color=colors_model["CNN"],
                        edgecolor="white")
    ax_bar.bar_label(bars_d, fmt="%.3f", padding=3, fontsize=9)
    ax_bar.bar_label(bars_c, fmt="%.3f", padding=3, fontsize=9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([k.replace("_", " ").title() for k in metric_keys],
                            fontsize=11)
    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_ylabel("Score (test set)", fontsize=11)
    ax_bar.set_title("Test-set Metric Comparison", fontsize=12)
    ax_bar.legend(fontsize=10)
    ax_bar.axhline(y=1.0, color="grey", linestyle="--", alpha=0.4)

    fig.suptitle("Dense NN vs CNN — Final Comparison",
                 fontsize=14, fontweight="bold")

    out = PLOTS_DIR / "final_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Final comparison plot → {out}")

    # Confusion matrices for both models
    plot_confusion_matrix(
        test_dense["confusion_matrix"],
        title=f"Dense NN — Test Set (F1={test_dense['f1_score']:.3f})",
        filename="final_dense_cm.png",
    )
    plot_confusion_matrix(
        test_cnn["confusion_matrix"],
        title=f"CNN — Test Set (F1={test_cnn['f1_score']:.3f})",
        filename="final_cnn_cm.png",
    )


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

#: Maps ``--person`` argument to the corresponding study function.
STUDY_MAP: Dict[int, object] = {
    1: study_learning_rate,   # CNN · learning rate
    2: study_batch_size,      # CNN · batch size
    3: study_dropout,         # CNN · dropout rate
    4: study_epochs,          # CNN · training epochs  ← replaced Dense depth
    5: study_image_size,      # CNN · image resolution
    6: compare_dense_vs_cnn,  # Dense NN vs CNN final comparison
}

STUDY_NAMES: Dict[int, str] = {
    1: "Learning Rate (Yogesh)",
    2: "Batch Size (Pradnya)",
    3: "Dropout Rate (Sakshee)",
    4: "Epochs (Kiran)",
    5: "Image Size (Adama)",
    6: "Dense NN vs CNN final comparison",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter study for Cable Defect Classifier",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--person",
        type=int,
        choices=list(STUDY_MAP.keys()),
        required=True,
        help=(
            "Which study to run:\n"
            "  1 = Learning Rate  (CNN)\n"
            "  2 = Batch Size     (CNN)\n"
            "  3 = Dropout Rate   (CNN)\n"
            "  4 = Epochs         (CNN)\n"
            "  5 = Image Size     (CNN)\n"
            "  6 = Dense NN vs CNN final comparison\n"
        ),
    )
    parser.add_argument(
        "--dataset_dir",
        default=DATASET_DIR,
        help=f"Path to dataset folder (default: {DATASET_DIR})",
    )
    args = parser.parse_args()

    # Allow overriding the dataset path at runtime.
    DATASET_DIR = args.dataset_dir

    print(f"\n  Running: {STUDY_NAMES[args.person]}")
    print(f"  MLflow:  {MLFLOW_URI}")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Seed:    {SEED}")
    print(f"  Patience:{PATIENCE}\n")

    STUDY_MAP[args.person]()  # type: ignore[operator]

    print(f"\n✓  Done! View results:")
    print(f"   mlflow ui --backend-store-uri {MLFLOW_URI}")
    print(f"   Then open http://127.0.0.1:5000")
    print(f"   Plots   → {PLOTS_DIR}/")
    print(f"   Results → {RESULTS_DIR}/")
