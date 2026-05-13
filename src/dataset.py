import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Creates and returns PyTorch DataLoaders for the training, validation, and test datasets.
    
    Args:
        data_dir (str): The root directory containing 'train', 'valid', and 'test' folders.
        batch_size (int): Number of images to process in a single batch.
        num_workers (int): Number of parallel processes to use for data loading.
        
    Returns:
        train_loader, valid_loader, test_loader, class_names
    """
    
    # 1. Define Data Augmentation for Training
    # We apply random transformations to make the model more robust.
    # If the model only sees perfectly centered leaves, it won't generalize well to real-world photos.
    train_transform = transforms.Compose([
        # Resize all images to 224x224 pixels (standard size for most CNNs like ResNet)
        transforms.Resize((224, 224)),
        
        # Randomly flip the image horizontally (left to right) with a 50% chance
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Randomly rotate the image by up to 20 degrees
        #transforms.RandomRotation(degrees=20),
        
        # Convert the image from a PIL Image format into a PyTorch Tensor (numbers between 0 and 1)
        transforms.ToTensor(),
        
        # Normalize the colors using ImageNet standards. 
        # This helps the neural network learn faster by keeping numbers small and centered around 0.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Define Data Processing for Validation and Testing
    # Note: We DO NOT use random flips or rotations here. We want to test the model 
    # on consistent, unaltered images to get an accurate measure of its performance.
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. Load the Datasets from Folders
    # PyTorch's ImageFolder automatically infers the class names from the folder structure!
    # For example, all images in data/train/Apple___healthy are labeled "Apple___healthy".
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'valid'), transform=test_transform)
    
    # The Kaggle test folder doesn't have class folders, just images in a 'test' subfolder.
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=test_transform)

    # 4. Create the DataLoaders
    # DataLoaders handle grouping the images into batches, shuffling them, and loading them in parallel.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # We don't need to shuffle validation or test data since we are just evaluating them.
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Get the list of disease names (class names)
    class_names = train_dataset.classes

    return train_loader, valid_loader, test_loader, class_names

# Example usage (if this file is run directly)
if __name__ == "__main__":
    # Since we are running the script from the ~/mscs folder, the path is simply 'data'
    data_dir = 'data'
    
    print("Testing data loaders...")
    train_loader, valid_loader, test_loader, classes = get_data_loaders(data_dir, batch_size=16)
    
    print(f"\nSuccess! Found {len(classes)} different plant disease classes.")
    print(f"First 5 classes: {classes[:5]}")
    
    # Get a single batch of images
    images, labels = next(iter(train_loader))
    print(f"\nBatch Image Shape: {images.shape}") # Should be [16, 3, 224, 224]
    print("Shape explained: [batch_size, color_channels, height, width]")