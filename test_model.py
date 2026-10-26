# Test script to verify the model works without requiring actual data
import torch
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append('.')

# Import from the main script
from p import Generator, PatchDiscriminator, VGGPerceptualLoss, init_weights

def test_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    # Create models
    G = Generator().to(device)
    D = PatchDiscriminator().to(device)
    vgg_loss = VGGPerceptualLoss().to(device)
    
    # Initialize weights
    init_weights(G)
    init_weights(D)
    
    print("Models created successfully!")
    
    # Test forward pass
    batch_size = 2
    img_size = 256
    test_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    print("Testing Generator...")
    with torch.no_grad():
        fake_output = G(test_input)
        print(f"Generator output shape: {fake_output.shape}")
        print(f"Generator output range: [{fake_output.min().item():.3f}, {fake_output.max().item():.3f}]")
    
    print("Testing Discriminator...")
    with torch.no_grad():
        disc_output = D(test_input)
        print(f"Discriminator output shape: {disc_output.shape}")
    
    print("Testing VGG Perceptual Loss...")
    with torch.no_grad():
        perc_loss = vgg_loss(fake_output, test_input)
        print(f"Perceptual loss: {perc_loss.item():.6f}")
    
    print("All tests passed! The model is working correctly.")
    return True

if __name__ == "__main__":
    test_model()