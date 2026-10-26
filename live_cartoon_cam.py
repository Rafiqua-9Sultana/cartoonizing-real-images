#!/usr/bin/env python3
"""
Real-time Camera Cartoonization using Trained GAN
Live video feed with cartoon effect applied in real-time
"""

import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from p import Generator, init_weights
import os

class LiveCartoonizer:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize the generator
        self.generator = Generator().to(self.device)
        init_weights(self.generator)
        
        # Load checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.generator.load_state_dict(checkpoint['G_state'])
                print(f"Loaded checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Could not load checkpoint: {e}")
                print("Using randomly initialized weights")
        else:
            print("No checkpoint found, using randomly initialized weights")
        
        self.generator.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)  # [-1, 1]
        ])
        
        # Post-processing to convert back to displayable format
        self.denormalize = transforms.Compose([
            transforms.Normalize([-1]*3, [2]*3),  # [-1,1] -> [0,1]
        ])
        
    def preprocess_frame(self, frame):
        """Convert OpenCV frame to tensor"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        # Apply transforms
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        return tensor
    
    def postprocess_output(self, tensor):
        """Convert tensor back to OpenCV frame"""
        # Denormalize
        tensor = self.denormalize(tensor)
        # Convert to numpy
        tensor = tensor.squeeze(0).cpu().clamp(0, 1)
        numpy_img = tensor.permute(1, 2, 0).numpy()
        # Convert to uint8
        numpy_img = (numpy_img * 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
        return frame_bgr
    
    def cartoonize_frame(self, frame):
        """Apply cartoonization to a single frame"""
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess_frame(frame)
            # Generate
            output_tensor = self.generator(input_tensor)
            # Postprocess
            cartoon_frame = self.postprocess_output(output_tensor)
            return cartoon_frame
    
    def run_camera(self, camera_id=0):
        """Main camera loop"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test camera first
        ret, test_frame = cap.read()
        if not ret:
            print("Error: Cannot read from camera")
            cap.release()
            return
        print(f"Camera working! Frame shape: {test_frame.shape}")
        
        print("Camera opened successfully!")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current cartoon frame")
        print("- Press 'r' to toggle between original and cartoon view")
        
        save_counter = 0
        show_cartoon = True
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Resize frame for consistent processing
                frame = cv2.resize(frame, (640, 480))
                
                if show_cartoon:
                    try:
                        # Apply cartoonization
                        cartoon_frame = self.cartoonize_frame(frame)
                        # Resize back to display size
                        cartoon_frame = cv2.resize(cartoon_frame, (640, 480))
                        
                        # Create side-by-side display
                        display_frame = np.hstack((frame, cartoon_frame))
                        window_title = "Live Camera - Original (Left) vs Cartoon (Right)"
                    except Exception as e:
                        print(f"Cartoonization error: {e}")
                        display_frame = frame
                        window_title = "Live Camera (Error - Showing Original)"
                else:
                    display_frame = frame
                    window_title = "Live Camera (Original Mode)"
                
                # Add status text
                if show_cartoon and display_frame.shape[1] > 640:  # Side by side mode
                    cv2.putText(display_frame, "Original", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Cartoon", (650, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    status_text = "Cartoon Mode" if show_cartoon else "Original Mode"
                    cv2.putText(display_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow(window_title, display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s') and show_cartoon:
                    # Save cartoon frame
                    filename = f"cartoon_capture_{save_counter:03d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved: {filename}")
                    save_counter += 1
                elif key == ord('r'):
                    # Toggle between original and cartoon
                    show_cartoon = not show_cartoon
                    print(f"Switched to {'Cartoon' if show_cartoon else 'Original'} mode")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")

def main():
    # Look for the most recent checkpoint
    checkpoint_dir = "./checkpoints"
    checkpoint_path = None
    
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if checkpoints:
            # Get the most recent checkpoint
            checkpoints.sort()
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
    
    # Create cartoonizer
    cartoonizer = LiveCartoonizer(checkpoint_path=checkpoint_path)
    
    # Start camera
    print("Starting live cartoon camera...")
    cartoonizer.run_camera()

if __name__ == "__main__":
    main()