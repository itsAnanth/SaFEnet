import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse

# If running as a script, we use relative/absolute imports appropriately
try:
    from model import get_resnet_feature_extractor
except ImportError:
    from src.model import get_resnet_feature_extractor

def predict_image(image_path, checkpoint_path, class_names=None):
    """
    Loads a saved checkpoint and runs inference on a single image.
    """
    if class_names is None:
        # Default for CIFAKE ImageFolder (alphabetical: FAKE, REAL)
        class_names = ['FAKE', 'REAL']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the model architecture
    model = get_resnet_feature_extractor(num_classes=len(class_names))
    model = model.to(device)
    
    # 2. Load the trained weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Define transforms mimicking the exact CIFAKE dataset artifacts
    transform = transforms.Compose([
        # Step 1: Crush down to 32x32 to match CIFAKE native resolution
        transforms.Resize((32, 32)),
        # Step 2: Scale up to 224x224 using the same interpolation the training dataloader used
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 4. Load & preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # 5. Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    predicted_class = class_names[predicted_idx.item()]
    conf_score = confidence.item() * 100
    
    print(f"--- Prediction Results ---")
    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {conf_score:.2f}%")
    
    # Also print the raw probabilities for all classes
    for i, cls_name in enumerate(class_names):
        print(f"  {cls_name} probability: {probabilities[0, i].item() * 100:.2f}%")
        
    return predicted_class, conf_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a custom image against the trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file to test")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the best checkpoint (.pt file)")
    
    args = parser.parse_args()
    predict_image(args.image, args.checkpoint)
