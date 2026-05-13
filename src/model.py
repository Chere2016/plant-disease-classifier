import torch
import torch.nn as nn
import torch.nn.functional as F

class PlantDiseaseCNN(nn.Module):
    """
    A custom Convolutional Neural Network (CNN) built from scratch to classify plant diseases.
    
    This architecture uses 3 Convolutional Blocks followed by Fully Connected layers.
    It expects input images of shape: [batch_size, 3 (channels), 224 (height), 224 (width)]
    """
    def __init__(self, num_classes=38):
        super(PlantDiseaseCNN, self).__init__()
        
        # --- BLOCK 1 ---
        # Input: [batch_size, 3, 224, 224]
        # We start by using 32 filters (feature detectors) to look for simple edges and colors.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # MaxPool reduces the image size by half. Output after pool: [batch_size, 32, 112, 112]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- BLOCK 2 ---
        # We increase filters to 64 to look for more complex shapes (like spots on leaves).
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Output after pool: [batch_size, 64, 56, 56]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- BLOCK 3 ---
        # We increase filters to 128 to look for very complex patterns (like specific disease textures).
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Output after pool: [batch_size, 128, 28, 28]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- FULLY CONNECTED CLASSIFIER ---
        # After flattening the 3D tensor into a 1D line, the size will be: 128 * 28 * 28 = 100,352
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        # Dropout randomly turns off 50% of the neurons during training to prevent memorization (overfitting)
        self.dropout = nn.Dropout(0.5)
        # Final layer outputs the prediction scores for our 38 plant disease classes
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        This defines how the image flows through the network.
        """
        # Pass through Block 1, apply ReLU activation (turns negative numbers to 0), then pool
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Pass through Block 2
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Pass through Block 3
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten the image from a 3D box into a 1D line of numbers
        x = torch.flatten(x, 1) # Flattens starting from the 1st dimension (keeping batch_size separate)
        
        # Pass through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Example usage (if this file is run directly)
if __name__ == "__main__":
    # Create a dummy batch of 4 random images (3 color channels, 224x224 pixels)
    dummy_input = torch.randn(4, 3, 224, 224)
    
    print("Testing the CNN Architecture...")
    model = PlantDiseaseCNN(num_classes=38)
    
    # Pass the fake images through the model
    output = model(dummy_input)
    
    print(f"Model successfully processed the input!")
    print(f"Output shape: {output.shape}") 
    print("Shape explained: [batch_size, num_classes] -> [4 images, 38 prediction scores]")
