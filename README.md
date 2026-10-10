# MVTec Cable Defect Binary Classification

An industry-standard deep learning project for binary classification of cable defects using the MVTec Cable dataset.

## Project Overview

This project implements a complete end-to-end pipeline for binary cable defect classification (OK vs NOT_OK). It includes data preparation, model training, evaluation, and visualization components.

**Models Implemented:**
- Dense Neural Network (fully connected layers)
- Convolutional Neural Network (CNN)

**Features:**
- Stratified train/val/test split
- Data augmentation (flips, rotations, color jitter)
- Early stopping and model checkpointing
- MLflow experiment tracking
- Comprehensive metrics and visualizations
- Production-level code quality

## Project Structure

```
Cable_Defect_Project/
├── src/
│   ├── config/              # Configuration module
│   │   └── config.py
│   ├── data/                # Data preparation
│   │   └── __init__.py
│   ├── datasets/            # Dataset loader
│   │   └── __init__.py
│   ├── models/              # Model definitions
│   │   ├── dense_model.py
│   │   ├── cnn_model.py
│   │   └── __init__.py
│   ├── training/            # Training pipeline
│   │   └── __init__.py
│   ├── evaluation/          # Evaluation and metrics
│   │   └── __init__.py
│   └── utils/               # Utilities and logging
│       └── __init__.py
│
├── prepare_dataset.py       # Data preparation script
├── train_dense.py           # Dense model training
├── train_cnn.py             # CNN model training
├── evaluate_model.py        # Model evaluation
│
├── dataset_binary/          # Binary dataset (OK, NOT_OK)
├── saved_models/            # Trained model checkpoints
├── logs/                    # Training logs
├── plots/                   # Visualization outputs
├── results/                 # Metrics and results
├── mlflow.db                  # MLflow tracking
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:
```bash
cd Cable_Defect_Project
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Prepare Dataset

Organize raw MVTec cable data into binary classes:

```bash
python prepare_dataset.py \
    --raw-data-dir data_raw/cable \
    --output-dir dataset_binary
```

This creates:
- `dataset_binary/OK/` - Contains good cables (train/good + test/good)
- `dataset_binary/NOT_OK/` - Contains all defect types

### Step 2: Train Dense Model

```bash
python train_dense.py \
    --batch-size 32 \
    --epochs 100 \
    --lr 1e-3 \
    --dropout 0.5 \
    --dataset-dir dataset_binary
```

**Arguments:**
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-3)
- `--dropout`: Dropout rate (default: 0.5)
- `--dataset-dir`: Path to binary dataset (default: dataset_binary)

**Output:**
- `saved_models/best_model.pth` - Best model checkpoint
- `saved_models/dense_model_final.pth` - Final model
- `plots/loss_curve.png`, `accuracy_curve.png` - Training history
- `plots/confusion_matrix.png`, `roc_curve.png` - Evaluation plots
- `results/metrics.json` - Performance metrics
- `logs/train_dense.log` - Training logs
- `mlflow.db` - MLflow experiment tracking

### Step 3: Train CNN Model

```bash
python train_cnn.py \
    --batch-size 32 \
    --epochs 100 \
    --lr 1e-3 \
    --dropout 0.5 \
    --dataset-dir dataset_binary
```

Similar arguments as dense model training.

### Step 4: Evaluate Model

Evaluate a trained model on the test set:

```bash
python evaluate_model.py \
    --model dense \
    --model-path saved_models/dense_model_best.pth \
    --dataset-dir dataset_binary \
    --batch-size 32
```

Or for CNN:
```bash
python evaluate_model.py \
    --model cnn \
    --model-path saved_models/cnn_model_best.pth \
    --dataset-dir dataset_binary
```

## Configuration

Core configuration is defined in `src/config/config.py`:

- **DataConfig**: Dataset paths, image size (128×128), train/val/test split ratios
- **TrainConfig**: Batch size, learning rate, number of epochs, early stopping parameters
- **ModelConfig**: Model architecture parameters (dropout, hidden sizes, etc.)
- **EvaluationConfig**: Paths for plots and results

## Model Architectures

### Dense Neural Network
- Flatten (128×128×3 → 49152)
- Dense(49152 → 512) + ReLU + Dropout
- Dense(512 → 256) + ReLU + Dropout
- Dense(256 → 64) + ReLU + Dropout
- Dense(64 → 1) + Sigmoid

### CNN
- Conv2d(3 → 32) + ReLU + MaxPool2d
- Conv2d(32 → 64) + ReLU + MaxPool2d
- Conv2d(64 → 128) + ReLU + MaxPool2d
- Flatten
- Dense(flat_size → 256) + ReLU + Dropout(0.5)
- Dense(256 → 1) + Sigmoid

## Data Augmentation

Training data augmentation includes:
- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization (ImageNet stats)

## Metrics & Evaluation

**Metrics tracked:**
- Accuracy
- Precision
- Recall
- F1 Score
- Loss (BCE)

**Visualizations generated:**
- Training/validation loss curves
- Training/validation accuracy curves
- Confusion matrix
- ROC curve with AUC

## Experiment Tracking

MLflow is used for experiment tracking:

```bash
mlflow ui
```

Then open http://localhost:5000 to view experiments, compare runs, and track metrics.

## Results

After training, check:
- `results/metrics.json` - Test metrics
- `plots/confusion_matrix.png` - Confusion matrix
- `plots/roc_curve.png` - ROC curve
- `logs/` - Training logs
- `mlflow.db` - MLflow experiment runs

## Requirements

- Python 3.8+
- PyTorch 2.0+
- scikit-learn
- matplotlib
- MLflow
- CUDA (optional, for GPU acceleration)

## Code Quality

The project follows production standards:
- Type hints for all functions
- Docstrings for modules and classes
- Proper error handling
- Logging throughout
- Modular, reusable components
- Configuration management
- Stratified data splitting

## Performance Tips

1. **GPU Training**: Ensure PyTorch is installed with CUDA support
2. **Data Loading**: Adjust `num_workers` based on CPU cores
3. **Batch Size**: Larger batches for faster training (if GPU memory allows)
4. **Early Stopping**: Prevents overfitting and saves training time
5. **Validation Split**: 15% validation set for robust evaluation

## Common Issues

**Model not loading:**
```python
# Ensure model architecture matches saved checkpoint
model = DenseModel(input_size=128*128*3, hidden_sizes=[512, 256, 64])
model.load_state_dict(torch.load('saved_models/best_model.pth'))
```

**Out of memory:**
- Reduce batch size: `--batch-size 16`
- Reduce image size in `src/config/config.py`

**Dataset not found:**
- Verify `dataset_binary/` directory exists
- Run `prepare_dataset.py` first

## References

- MVTec Cable Dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- MLflow: https://mlflow.org/docs/latest/index.html

## License

Academic use only. For other uses, refer to MVTec AD dataset license.

---

**Author**: Deep Learning Project
**Last Updated**: June 2026
