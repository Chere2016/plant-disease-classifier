# Plant Disease Classifier

A custom Convolutional Neural Network (CNN) built from scratch in PyTorch to classify 38 classes of plant leaf diseases. Features a professional MLOps training pipeline with Weights & Biases logging, L2 regularization, and dynamic learning rate scheduling, achieving **98.7% test accuracy**.

## Features
- **Custom Architecture**: A highly optimized CNN designed specifically for 224x224 RGB leaf images across 38 categories.
- **Advanced Regularization**: Uses L2 Weight Decay (`1e-4`) to aggressively combat overfitting on the training set.
- **Dynamic Learning Rate**: Implements PyTorch's `ReduceLROnPlateau` scheduler to mathematically find the absolute minimum loss.
- **Cloud MLOps**: Fully integrated with Weights & Biases (`wandb`) for real-time dashboard monitoring of train/validation curves.

## Directory Structure
- `src/`: Python source code containing the dataset loader (`dataset.py`), model architecture (`model.py`), and the core training loop (`train.py`).
- `configs/`: YAML configuration files defining hyperparameters for different experiments.
- `scripts/`: Bash scripts containing `nohup` commands to run experiments safely in the background on specific GPUs.
- `notebooks/`: Jupyter Notebooks used for data exploration and final model evaluation.

## How to Run

1. **Setup Environment**:
Make sure you have your dependencies (PyTorch, wandb, torchvision) installed in a conda environment.

2. **Train the Model**:
Choose a bash script from the `scripts/` folder. Ensure you configure your `CUDA_VISIBLE_DEVICES` correctly in the script before running:
```bash
./scripts/run_train_exp4.sh
```

3. **Evaluate**:
Launch `notebooks/evaluate.ipynb` to load your best `.pth` model weights and test predictions on raw, unseen images.

## Performance
- **Experiment 4 (100 Epochs)**: Achieved a final, stable validation accuracy of **98.72%**. Inference on unseen test images yielded predictions with >99.9% confidence.
