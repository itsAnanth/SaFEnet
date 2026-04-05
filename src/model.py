import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SaFENet(nn.Module):
    def __init__(self, num_classes=2):
        super(SaFENet, self).__init__()
        
        # 1. ResNet Backbone
        weights = models.ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights=weights)
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Unfreeze the final spatial block (layer4) for fine-tuning
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
            
        resnet_out_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() # Strip off the original FC
        
        # 2. Gradient Edge CNN branch
        # This will process the grayscale sobel gradient magnitudes
        self.grad_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        grad_features = 32
        
        # 3. Frequency Spectrum CNN branch
        # This will process the log-scaled Fourier transform magnitude
        self.freq_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        freq_features = 32
        
        total_features = resnet_out_features + grad_features + freq_features
        
        # 4. Feature Attention Gate
        # Dynamically weights the importance of semantic, gradient, and frequency features
        self.attention_gate = nn.Sequential(
            nn.Linear(total_features, total_features // 4),
            nn.ReLU(),
            nn.Linear(total_features // 4, total_features),
            nn.Sigmoid()
        )
        
        # 5. Fusion Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Setup static Sobel filters
        sobel_x = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]]).float()
        sobel_y = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]]).float()
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def get_grayscale(self, x):
        # Convert RGB inputs (nominally normalized) to grayscale
        # Using BT.601 standard photometric weights
        return 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        
    def compute_gradients(self, gray):
        # Take grayscale, compute Sobel edges
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
        return magnitude
        
    def compute_frequency_spectrum(self, gray):
        # Compute 2D Fast Fourier Transform
        fft2 = torch.fft.fft2(gray)
        # Shift the zero-frequency component to the center of the spectrum
        shifted_fft = torch.fft.fftshift(fft2, dim=(-2, -1))
        # Get magnitude and apply log scale to compress dynamic range
        magnitude = torch.abs(shifted_fft)
        log_magnitude = torch.log(magnitude + 1e-6)
        return log_magnitude

    def forward(self, x):
        # Forward pass main image through resnet
        f_res = self.resnet(x)
        
        # Calculate grayscale map once to feed both custom auxiliary branches
        gray = self.get_grayscale(x)
        
        # 1. Compute dynamic gradients and push them through CNN branch
        grad_img = self.compute_gradients(gray)
        f_grad = self.grad_cnn(grad_img)
        
        # 2. Compute frequency spectrum and push through Frequency CNN branch
        freq_img = self.compute_frequency_spectrum(gray)
        f_freq = self.freq_cnn(freq_img)
        
        # Concatenate spatial ResNet embeddings, local edge gradients, and global periodic frequencies
        f_cat = torch.cat([f_res, f_grad, f_freq], dim=1)
        
        # Apply Attention Gating
        # This will emphasize useful features and suppress noisy ones for this specific image
        attention_weights = self.attention_gate(f_cat)
        f_fused = f_cat * attention_weights
        
        # Final classification
        out = self.fc(f_fused)
        return out

def get_resnet_feature_extractor(num_classes=2):
    # Returns the new Gradient-Fused ResNet model (SaFENet)
    return SaFENet(num_classes=num_classes)

def get_baseline_resnet(num_classes=2):
    """
    Returns a standard ResNet18 baseline model.
    It uses the same exact backbone logic (frozen trunk, unfrozen layer4) 
    so you can do an apples-to-apples comparison against SaFENet.
    """
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # Freeze the backbone
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze layer4 to match SaFENet's capacity
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Replace the final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    
    return model

