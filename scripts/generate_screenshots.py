import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# Set CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Add src to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from model import PlantDiseaseCNN
from dataset import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get classes
data_dir = os.path.join(os.path.dirname(__file__), '../data')
_, _, _, classes = get_data_loaders(data_dir, batch_size=1)
num_classes = len(classes)

# Load model
model = PlantDiseaseCNN(num_classes=num_classes)
model_path = os.path.join(os.path.dirname(__file__), '../models/plant_disease_cnn_exp4_100epochs_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def save_prediction_screenshot(image_path, output_path, classes, top_k=3):
    img = Image.open(image_path).convert('RGB')
    img_tensor = eval_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        
    probabilities = F.softmax(outputs, dim=1)[0] * 100
    top_prob, top_catid = torch.topk(probabilities, top_k)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.imshow(img)
    ax.axis('off')
    
    pred_text = "--- Top Predictions ---\n"
    for i in range(top_k):
        class_name = classes[top_catid[i]].replace('___', ' ').replace('_', ' ')
        score = top_prob[i].item()
        pred_text += f"{i+1}. {class_name} ({score:.2f}%)\n"
        
    ground_truth = os.path.basename(image_path).split('.')[0]
    pred_text += f"\nGround Truth (from filename):\n{ground_truth}"
        
    # Add text to the right of the image
    plt.text(1.05, 0.5, pred_text, transform=ax.transAxes, fontsize=11,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='#e6f3ff', alpha=0.9, edgecolor='#b3d9ff'))
             
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()

# Generate screenshots
assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
test_dir = os.path.join(os.path.dirname(__file__), '../data/test/test')

save_prediction_screenshot(os.path.join(test_dir, 'PotatoHealthy2.JPG'), os.path.join(assets_dir, 'prediction_potato.png'), classes)
save_prediction_screenshot(os.path.join(test_dir, 'TomatoEarlyBlight1.JPG'), os.path.join(assets_dir, 'prediction_tomato.png'), classes)
save_prediction_screenshot(os.path.join(test_dir, 'AppleCedarRust1.JPG'), os.path.join(assets_dir, 'prediction_apple.png'), classes)

print("Screenshots generated successfully!")
