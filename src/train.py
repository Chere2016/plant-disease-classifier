import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import yaml
import argparse
import wandb
import random
import numpy as np
from tqdm import tqdm
# Import our custom dataset and model
from dataset import get_data_loaders
from model import PlantDiseaseCNN

def train_model(config_path):
    # --- 0. Set Random Seed for Reproducibility ---
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    # --- 1. Load Configuration from YAML ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print("Loaded Configuration:", config)
    
    # --- 2. Initialize Weights & Biases (WandB) ---
    # This automatically syncs all our hyperparameters to the cloud dashboard!
    wandb.init(
        project="plant-disease-classification", 
        name=config['experiment_name'],
        config=config
    )

    # --- 3. Hardware Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Training on device: {device}")

    # --- 4. Get Data ---
    print("Loading data...")
    train_loader, valid_loader, test_loader, classes = get_data_loaders(
        data_dir=config['data_dir'], 
        batch_size=config['batch_size']
    )
    num_classes = len(classes)
    print(f"Data loaded successfully! Training on {num_classes} classes.")
    
    # --- 5. Initialize Model, Loss Function, and Optimizer ---
    model = PlantDiseaseCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    # Add Weight Decay (L2 Penalty) to penalize large weights and prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    
    # Add a Learning Rate Scheduler to automatically lower the LR if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Log model gradients and architecture to wandb
    wandb.watch(model, criterion, log="all", log_freq=10)

    # --- 6. The Main Training Loop ---
    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        
        # Wrap train_loader with tqdm to show a live progress bar
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        
        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            # Wrap valid_loader with tqdm as well
            for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", leave=False):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                
                # Calculate validation loss for this batch
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs, 1)
                
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / len(valid_loader.dataset)
        epoch_acc = (correct_predictions / total_predictions) * 100
        epoch_time = time.time() - start_time
        
        # Step the scheduler based on the validation loss
        scheduler.step(epoch_val_loss)
        
        # Get current learning rate to log it
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.6f} | Train Loss: {epoch_loss:.4f} | Valid Loss: {epoch_val_loss:.4f} | Valid Acc: {epoch_acc:.2f}% | Time: {epoch_time:.1f}s")
        
        # --- LOG METRICS TO WANDB ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "valid_loss": epoch_val_loss,
            "valid_accuracy": epoch_acc,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time
        })

    # --- 7. Save the trained model ---
    os.makedirs('models', exist_ok=True)
    save_path = f"models/{config['experiment_name']}_model.pth"
    torch.save(model.state_dict(), save_path)
    
    # Optional: Save the model file directly to wandb so you can download it from the cloud later
    wandb.save(save_path)
    
    print(f"\nTraining Complete! Model saved successfully to: {save_path}")
    wandb.finish()

if __name__ == "__main__":
    # Setup argparse to allow passing the config file via command line
    parser = argparse.ArgumentParser(description="Train the Plant Disease CNN")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to the YAML config file')
    args = parser.parse_args()
    
    train_model(args.config)
