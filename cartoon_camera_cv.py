#!/usr/bin/env python3
"""
Real-time Camera with Cartoon Filter using OpenCV
This uses traditional computer vision techniques for immediate cartoon effect
"""

import cv2
import numpy as np

class CartoonFilter:
    def __init__(self):
        pass
    
    def apply_ghibli_cartoon(self, frame):
        """Apply Studio Ghibli anime style cartoon effect"""
        # Step 1: Bilateral filter for smooth regions while preserving edges
        smooth = cv2.bilateralFilter(frame, 15, 80, 80)
        smooth = cv2.bilateralFilter(smooth, 15, 80, 80)
        
        # Step 2: Create clean, anime-style edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 7)
        
        # Use adaptive threshold for clean lines
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 7, 7)
        
        # Clean up edges
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # Convert edges to 3-channel
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Step 3: Ghibli-style color quantization (limited palette)
        data = smooth.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = 8  # Limited colors for anime look
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized = quantized_data.reshape(smooth.shape)
        
        # Step 4: Combine quantized image with edges
        cartoon = cv2.bitwise_and(quantized, edges)
        
        # Step 5: Ghibli-style color enhancement
        # Boost contrast for that crisp anime look
        cartoon = cv2.convertScaleAbs(cartoon, alpha=1.3, beta=10)
        
        # Enhance colors like Ghibli films
        hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
        
        # Boost saturation significantly for vibrant anime colors
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.4)
        
        # Increase brightness for that "anime glow"
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.2)
        
        cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Step 6: Apply slight blur to soften harsh edges (Ghibli style)
        cartoon = cv2.GaussianBlur(cartoon, (3, 3), 0)
        
        return cartoon

    def apply_professional_cartoon(self, frame):
        """Apply high-quality realistic cartoon style like professional digital art"""
        # Step 1: Very gentle smoothing to preserve facial features
        smooth = cv2.bilateralFilter(frame, 9, 60, 60)
        smooth = cv2.bilateralFilter(smooth, 9, 60, 60)
        
        # Step 2: Enhance the image quality first
        # Increase contrast and brightness slightly
        enhanced = cv2.convertScaleAbs(smooth, alpha=1.15, beta=10)
        
        # Step 3: Very subtle color smoothing (not aggressive quantization)
        # Use much more colors to maintain realism
        data = enhanced.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        k = 40  # Much more colors for realistic look
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        cartoon = quantized_data.reshape(enhanced.shape)
        
        # Step 4: Very minimal edge enhancement (almost invisible)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Create very subtle edges
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 9, 10)
        
        # Make edges extremely soft and barely visible
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        edges = cv2.dilate(edges, np.ones((1,1), np.uint8), iterations=1)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        
        # Convert to 3-channel with very low intensity
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges = edges.astype(np.float32) / 255.0
        
        # Step 5: Barely blend edges (95% smooth image, 5% edge effect)
        cartoon_float = cartoon.astype(np.float32)
        cartoon = cartoon_float * (edges * 0.05 + 0.95)
        cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
        
        # Step 6: Professional color enhancement
        hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
        
        # Very gentle saturation boost
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)
        
        # Slight warmth
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.02, 0, 255)
        
        cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Step 7: Final very gentle smoothing
        cartoon = cv2.bilateralFilter(cartoon, 3, 20, 20)
        
        return cartoon
    
    def apply_super_clear_cartoon(self, frame):
        """Apply clear cartoon filter with bright, vivid colors"""
        # Step 1: Strong smoothing for clean regions
        smooth = cv2.bilateralFilter(frame, 12, 100, 100)
        smooth = cv2.bilateralFilter(smooth, 12, 100, 100)
        
        # Step 2: Gentle edge detection to avoid darkness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use gentler edge detection
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 11, 8)
        
        # Dilate edges to make them thicker and cleaner
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Convert to 3-channel
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Step 3: Color quantization for clear, distinct colors
        data = smooth.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        k = 10  # Good balance of colors
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized = quantized_data.reshape(smooth.shape)
        
        # Step 4: Blend instead of harsh combination to avoid darkness
        # Convert edges to float for blending
        edges_float = edges.astype(np.float32) / 255.0
        quantized_float = quantized.astype(np.float32)
        
        # Blend the images (multiply effect but preserve brightness)
        cartoon = quantized_float * edges_float
        cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
        
        # Step 5: Brighten and enhance
        # Increase brightness and contrast
        cartoon = cv2.convertScaleAbs(cartoon, alpha=1.3, beta=20)
        
        # Boost saturation for vivid colors
        hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.4)  # Increase saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)  # Slight brightness boost
        cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return cartoon
    
    def apply_cartoon_filter(self, frame):
        """Apply high-quality cartoon effect with clear, sharp details"""
        # Step 1: Apply strong bilateral filter for smooth regions
        smooth = cv2.bilateralFilter(frame, 15, 120, 120)
        smooth = cv2.bilateralFilter(smooth, 15, 120, 120)
        
        # Step 2: Create sharp, clean edge mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        
        # Use adaptive threshold for cleaner edges
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 9, 9)
        
        # Morphological operations to clean up edges
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # Convert to 3-channel
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Step 3: Color quantization for cleaner colors
        data = smooth.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 12  # Number of colors
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized_image = quantized_data.reshape(smooth.shape)
        
        # Step 4: Combine quantized image with sharp edges
        cartoon = cv2.bitwise_and(quantized_image, edges)
        
        # Step 5: Enhance contrast and saturation
        cartoon = cv2.convertScaleAbs(cartoon, alpha=1.2, beta=10)
        
        # Convert to HSV for saturation boost
        hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.4)  # Increase saturation
        cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return cartoon
    
    def apply_anime_filter(self, frame):
        """Apply high-quality anime-style filter with clear details"""
        # Strong bilateral filtering for very smooth regions
        smooth = cv2.bilateralFilter(frame, 20, 150, 150)
        smooth = cv2.bilateralFilter(smooth, 20, 150, 150)
        
        # Sharp edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Use Canny for very crisp edges
        edges = cv2.Canny(gray_blur, 50, 150)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
        edges = 255 - edges  # Invert for white background
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Aggressive color quantization for anime look
        data = smooth.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        k = 6  # Very few colors for anime look
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized = quantized_data.reshape(smooth.shape)
        
        # Combine with sharp edges
        cartoon = cv2.bitwise_and(quantized, edges)
        
        # Boost colors for anime vibrancy
        cartoon = cv2.convertScaleAbs(cartoon, alpha=1.3, beta=15)
        
        # Increase saturation dramatically
        hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.6)
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.1)  # Slight brightness boost
        cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return cartoon
    
    def apply_disney_cartoon(self, frame):
        """Apply Disney/Pixar-style 3D cartoon filter"""
        # Step 1: Aggressive smoothing for porcelain-like skin
        smooth = cv2.bilateralFilter(frame, 20, 150, 150)
        smooth = cv2.bilateralFilter(smooth, 20, 150, 150)
        smooth = cv2.bilateralFilter(smooth, 15, 120, 120)  # Triple smoothing
        
        # Step 2: Face-like color enhancement (skin tones)
        # Convert to LAB for better skin tone processing
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
        
        # Enhance lightness and reduce blemishes
        l, a, b = cv2.split(lab)
        l = cv2.convertScaleAbs(l, alpha=1.2, beta=10)  # Brighter
        
        # Smooth A and B channels for even skin tone
        a = cv2.bilateralFilter(a, 10, 50, 50)
        b = cv2.bilateralFilter(b, 10, 50, 50)
        
        lab_enhanced = cv2.merge([l, a, b])
        smooth = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Step 3: Minimal edge detection (Disney style has soft edges)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Very soft edge detection - only major features
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 15, 10)
        
        # Soften edges even more
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        edges = cv2.convertScaleAbs(edges, alpha=0.6)  # Make edges lighter
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Step 4: Color quantization with Disney-like palette
        data = smooth.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = 6  # Fewer colors for Disney look
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized = quantized_data.reshape(smooth.shape)
        
        # Step 5: Soft blending (no harsh edges like Disney)
        edges_soft = edges.astype(np.float32) / 255.0
        cartoon = quantized.astype(np.float32) * (edges_soft * 0.3 + 0.7)  # Very soft edge influence
        cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
        
        # Step 6: Disney-style color enhancement
        # Increase warmth and vibrancy
        cartoon = cv2.convertScaleAbs(cartoon, alpha=1.15, beta=15)
        
        # Enhance colors like Disney animation
        hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
        
        # Boost saturation for vibrant Disney colors
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        
        # Slight brightness boost for that "glow"
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        
        cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Step 7: Final Disney-style glow effect
        # Create a soft glow
        glow = cv2.GaussianBlur(cartoon, (15, 15), 0)
        cartoon = cv2.addWeighted(cartoon, 0.8, glow, 0.2, 0)
        
        return cartoon

    def apply_bright_cartoon(self, frame):
        """Apply bright, colorful cartoon filter"""
        # Heavy bilateral filtering for smooth cartoon look
        smooth = cv2.bilateralFilter(frame, 15, 80, 80)
        smooth = cv2.bilateralFilter(smooth, 15, 80, 80)
        
        # Very gentle edge detection
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 13, 7)
        
        # Make edges thicker
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Color quantization with more colors to stay bright
        data = smooth.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 8
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized = quantized_data.reshape(smooth.shape)
        
        # Soft combination to preserve brightness
        edges_norm = edges.astype(np.float32) / 255.0
        cartoon = quantized.astype(np.float32) * (edges_norm * 0.7 + 0.3)
        cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
        
        # Boost brightness and colors
        cartoon = cv2.convertScaleAbs(cartoon, alpha=1.2, beta=25)
        
        # Enhance colors
        hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
        cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return cartoon

    def apply_oil_painting_filter(self, frame):
        """Apply oil painting effect"""
        # Use cv2.xphoto if available, otherwise use bilateral filter
        try:
            # Oil painting effect (if opencv-contrib is installed)
            result = cv2.xphoto.oilPainting(frame, 7, 1)
        except:
            # Fallback to bilateral filter
            result = cv2.bilateralFilter(frame, 9, 75, 75)
            result = cv2.bilateralFilter(result, 9, 75, 75)
        
        return result

class LiveCartoonCamera:
    def __init__(self):
        self.filter = CartoonFilter()
        self.current_filter = 0  # 0: ghibli, 1: professional, 2: bright cartoon, etc.
        self.filter_names = ["Ghibli Style", "Professional", "Bright Cartoon", "Super Clear", "Cartoon", "Anime", "Oil Painting"]
    
    def run_camera(self, camera_id=0):
        """Main camera loop"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Camera opened successfully!")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        print("- Press 'f' to switch filter (Ghibli Style/Professional/Bright/Clear/Cartoon/Anime/Oil)")
        print("- Press 'r' to toggle between original and filtered view")
        
        save_counter = 0
        show_filter = True
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Resize frame
                frame = cv2.resize(frame, (640, 480))
                
                if show_filter:
                    # Apply selected filter
                    if self.current_filter == 0:
                        filtered_frame = self.filter.apply_ghibli_cartoon(frame)
                    elif self.current_filter == 1:
                        filtered_frame = self.filter.apply_professional_cartoon(frame)
                    elif self.current_filter == 2:
                        filtered_frame = self.filter.apply_bright_cartoon(frame)
                    elif self.current_filter == 3:
                        filtered_frame = self.filter.apply_super_clear_cartoon(frame)
                    elif self.current_filter == 4:
                        filtered_frame = self.filter.apply_cartoon_filter(frame)
                    elif self.current_filter == 5:
                        filtered_frame = self.filter.apply_anime_filter(frame)
                    else:
                        filtered_frame = self.filter.apply_oil_painting_filter(frame)
                    
                    # Create side-by-side display
                    display_frame = np.hstack((frame, filtered_frame))
                    window_title = f"Live Camera - Original vs {self.filter_names[self.current_filter]}"
                    
                    # Add labels
                    cv2.putText(display_frame, "Original", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(display_frame, self.filter_names[self.current_filter], (650, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    display_frame = frame
                    window_title = "Live Camera - Original Mode"
                    cv2.putText(display_frame, "Original Mode", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow(window_title, display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"capture_{save_counter:03d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved: {filename}")
                    save_counter += 1
                elif key == ord('f'):
                    # Switch filter
                    self.current_filter = (self.current_filter + 1) % 7
                    print(f"Switched to {self.filter_names[self.current_filter]} filter")
                elif key == ord('r'):
                    # Toggle between original and filtered
                    show_filter = not show_filter
                    mode = "Filtered" if show_filter else "Original"
                    print(f"Switched to {mode} mode")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")

def main():
    print("Starting Live Cartoon Camera with Traditional Filters...")
    camera = LiveCartoonCamera()
    camera.run_camera()

if __name__ == "__main__":
    main()