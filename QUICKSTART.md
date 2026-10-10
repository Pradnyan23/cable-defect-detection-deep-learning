# Quick Start Guide

## Complete Project Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Organize MVTec cable data into binary classes (OK/NOT_OK):
```bash
python prepare_dataset.py
```

This will:
- Copy train/good + test/good → dataset_binary/OK/
- Copy all defect types → dataset_binary/NOT_OK/

### 3. Train Dense Model
```bash
python train_dense.py --epochs 50 --lr 1e-3 --batch-size 32
```

### 4. Train CNN Model
```bash
python train_cnn.py --epochs 50 --lr 1e-3 --batch-size 32
```

### 5. Evaluate Models
```bash
python evaluate_model.py --model dense --model-path saved_models/dense_model_best.pth
python evaluate_model.py --model cnn --model-path saved_models/cnn_model_best.pth
```

## Output Structure

After running all scripts, you'll have:

```
Cable_Defect_Project/
├── dataset_binary/          # Binary dataset (OK, NOT_OK)
├── saved_models/            # Trained model checkpoints
│   ├── best_model.pth       # Best checkpoint during training
│   ├── dense_model_final.pth
│   └── cnn_model_final.pth
├── logs/                    # Training logs
│   ├── train_dense.log
│   ├── train_cnn.log
│   └── prepare_dataset.log
├── plots/                   # Visualizations
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── results/                 # Metrics
│   └── metrics.json
└── mlflow.db                  # MLflow tracking
```

## View MLflow Dashboard

```bash
mlflow ui
```

Then open http://localhost:5000 to see:
- Training history
- Hyperparameter comparisons
- Metrics for each run

## Project Architecture

### Core Modules

**src/config/config.py**
- DataConfig: Dataset paths, splits, image size
- TrainConfig: Batch size, LR, epochs, early stopping
- ModelConfig: Model hyperparameters

**src/datasets/__init__.py**
- CableDataset: PyTorch Dataset class
- get_transforms(): Data augmentation pipeline
- create_dataloaders(): Creates train/val/test loaders with stratified split

**src/models/**
- dense_model.py: 3-layer FC network (49152→512→256→64→1)
- cnn_model.py: 3-layer CNN with MaxPool (32→64→128 channels)

**src/training/__init__.py**
- Trainer: Training loop with early stopping
- EarlyStopping: Stops training after val loss plateaus

**src/evaluation/__init__.py**
- Evaluator: Compute metrics, generate plots (confusion matrix, ROC)
- Generates: loss/accuracy curves, classification reports

**src/utils/__init__.py**
- setup_logging(): Configure logging to file + console
- create_directories(): Ensure output directories exist

### Data Flow

```
Raw Data (data_raw/cable)
    ↓
prepare_dataset.py
    ↓
Binary Dataset (dataset_binary/OK, NOT_OK)
    ↓
DataLoader (Stratified 70/15/15 split)
    ↓
Train/Val/Test
    ↓
Model (Dense or CNN)
    ↓
Trainer (with Early Stopping)
    ↓
Evaluation + Plots + MLflow Logging
```

## Key Features

✓ **Production-grade code**: Type hints, docstrings, error handling
✓ **Data augmentation**: Flips, rotations, color jitter
✓ **Early stopping**: Prevents overfitting
✓ **Stratified split**: Balanced class distribution
✓ **MLflow integration**: Track experiments automatically
✓ **Multiple visualizations**: Loss curves, confusion matrix, ROC
✓ **Modular design**: Easy to extend with new models

## Typical Training Metrics

**Dense Model**
- Accuracy: ~88-92%
- F1 Score: ~0.85-0.90
- Training time: ~2-5 minutes per epoch

**CNN Model**
- Accuracy: ~92-96%
- F1 Score: ~0.90-0.95
- Training time: ~3-8 minutes per epoch

## Tips for Best Results

1. **GPU Acceleration**: Install PyTorch with CUDA
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Hyperparameter Tuning**: Modify in config.py
   ```python
   batch_size = 64  # Larger for faster training
   learning_rate = 1e-4  # Smaller for fine-tuning
   num_epochs = 200  # More for better accuracy
   ```

3. **Monitor Training**: Watch logs in real-time
   ```bash
   tail -f logs/train_dense.log
   ```

4. **Compare Models**: Use MLflow dashboard
   ```bash
   mlflow ui
   ```

## Troubleshooting

**Q: Out of memory error?**
A: Reduce batch size: `--batch-size 16`

**Q: Dataset not found?**
A: Run `python prepare_dataset.py` first

**Q: Model not loading?**
A: Check model path and architecture match

**Q: Slow training?**
A: Enable GPU or reduce image size in config

---

For detailed documentation, see README.md
