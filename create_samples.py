# Create sample images for testing
from PIL import Image
import numpy as np
import os

def create_sample_images():
    # Create data/real directory if it doesn't exist
    os.makedirs("data/real", exist_ok=True)
    
    # Create a few sample images
    for i in range(5):
        # Create a random color image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Add some patterns to make it more interesting
        x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
        pattern = (np.sin(10 * x) * np.cos(10 * y) * 127 + 128).astype(np.uint8)
        
        img_array[:, :, 0] = (img_array[:, :, 0] * 0.7 + pattern * 0.3).astype(np.uint8)
        
        # Convert to PIL Image and save
        img = Image.fromarray(img_array)
        img.save(f"data/real/sample_{i}.jpg")
        print(f"Created sample_{i}.jpg")

if __name__ == "__main__":
    create_sample_images()
    print("Sample images created successfully!")