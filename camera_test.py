import cv2
import numpy as np

def test_camera():
    """Test different camera indices to find the working one"""
    print("Testing camera access...")
    
    for camera_id in range(5):  # Test cameras 0-4
        print(f"\nTrying camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            print(f"Camera {camera_id} opened successfully!")
            
            # Test reading a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Camera {camera_id} can read frames - Shape: {frame.shape}")
                
                # Show a test frame for 3 seconds
                cv2.imshow(f"Test Camera {camera_id}", frame)
                cv2.waitKey(3000)  # Show for 3 seconds
                cv2.destroyAllWindows()
                
                cap.release()
                return camera_id
            else:
                print(f"Camera {camera_id} cannot read frames")
        else:
            print(f"Camera {camera_id} failed to open")
        
        cap.release()
    
    print("\nNo working camera found!")
    return None

def simple_camera_test(camera_id=0):
    """Simple camera display without any processing"""
    print(f"Opening camera {camera_id} for live view...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camera opened! Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Cannot read frame")
            break
        
        # Add some text to confirm it's working
        cv2.putText(frame, "Normal Camera Test - Press 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Normal Camera Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    print("=== Camera Diagnostic Tool ===")
    
    # First test which cameras work
    working_camera = test_camera()
    
    if working_camera is not None:
        print(f"\nFound working camera: {working_camera}")
        print("Starting simple camera test...")
        simple_camera_test(working_camera)
    else:
        print("\nNo cameras detected. Check if:")
        print("1. Camera is connected")
        print("2. Camera drivers are installed") 
        print("3. No other application is using the camera")
        print("4. Camera permissions are granted")