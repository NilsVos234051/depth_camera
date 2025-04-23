#!/usr/bin/env python3
"""
Person Detection with Distance Calculation

This script extends the camera_setup.py functionality to:
1. Detect people using OpenCV DNN with MobileNet SSD
2. Calculate the distance of detected people using depth data
3. Visualize results with bounding boxes and distance information
4. Highlight people within a specified distance threshold

Requirements:
- pyrealsense2
- numpy
- opencv-python
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import sys
import argparse
import os

# Distance threshold in meters
DEFAULT_DISTANCE_THRESHOLD = 2.0  # 2 meters

def print_camera_info(device):
    """Print basic information about the connected camera."""
    print("\n" + "="*50)
    print("CAMERA INFORMATION")
    print("="*50)
    
    try:
        # Get device information
        print(f"Name: {device.get_info(rs.camera_info.name)}")
        print(f"Serial Number: {device.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware Version: {device.get_info(rs.camera_info.firmware_version)}")
        
        # Try to get additional information if available
        try:
            print(f"USB Type: {device.get_info(rs.camera_info.usb_type_descriptor)}")
        except:
            pass
            
        try:
            print(f"Product ID: {device.get_info(rs.camera_info.product_id)}")
        except:
            pass
            
        try:
            print(f"Camera Category: {device.get_info(rs.camera_info.product_line)}")
        except:
            pass
            
    except Exception as e:
        print(f"Error retrieving camera information: {e}")
    
    print("="*50 + "\n")

def load_person_detection_model():
    """
    Load the MobileNet SSD model for person detection.
    
    Returns:
        net: The loaded model
        None if loading fails
    """
    try:
        # Download model files if they don't exist
        model_file = "MobileNetSSD_deploy.caffemodel"
        config_file = "MobileNetSSD_deploy.prototxt"
        
        # Check if model files exist, if not download them
        if not os.path.exists(model_file) or not os.path.exists(config_file):
            print("Downloading model files...")
            
            # Download prototxt
            if not os.path.exists(config_file):
                prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
                os.system(f"curl -o {config_file} {prototxt_url}")
            
            # Download model
            if not os.path.exists(model_file):
                model_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
                os.system(f"curl -L -o {model_file} {model_url}")
        
        # Load the model
        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        print("Person detection model loaded successfully")
        return net
    except Exception as e:
        print(f"Error loading person detection model: {e}")
        return None

def detect_people(net, frame):
    """
    Detect people in the given frame using MobileNet SSD.
    
    Args:
        net: The loaded model
        frame: RGB image frame
        
    Returns:
        List of detections (bounding boxes) for people
    """
    if net is None:
        return []
    
    # Get frame dimensions
    (h, w) = frame.shape[:2]
    
    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 
        0.007843, 
        (300, 300), 
        127.5
    )
    
    # Pass the blob through the network
    net.setInput(blob)
    detections = net.forward()
    
    # Filter detections for people (class ID 15 in MobileNet SSD)
    people_detections = []
    
    for i in range(detections.shape[2]):
        # Extract the confidence
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.5:
            # Extract the class ID
            class_id = int(detections[0, 0, i, 1])
            
            # Class ID 15 is for person in MobileNet SSD
            if class_id == 15:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Add detection to the list
                people_detections.append({
                    'box': (startX, startY, endX, endY),
                    'confidence': confidence
                })
    
    return people_detections

def calculate_distance(depth_frame, x1, y1, x2, y2):
    """
    Calculate the distance to a person based on the depth frame and bounding box.
    
    Args:
        depth_frame: Depth frame from RealSense
        x1, y1, x2, y2: Bounding box coordinates
        
    Returns:
        Distance in meters
    """
    # Calculate center point of the bounding box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # Get a small region around the center to average depth values
    region_size = 5
    x_min = max(0, center_x - region_size)
    x_max = min(depth_frame.shape[1] - 1, center_x + region_size)
    y_min = max(0, center_y - region_size)
    y_max = min(depth_frame.shape[0] - 1, center_y + region_size)
    
    # Extract depth values in the region
    depth_region = depth_frame[y_min:y_max, x_min:x_max]
    
    # Filter out zero values (no depth data)
    valid_depths = depth_region[depth_region > 0]
    
    if len(valid_depths) == 0:
        return None  # No valid depth data
    
    # Calculate median depth to reduce noise
    median_depth = np.median(valid_depths)
    
    # Convert to meters (RealSense depth values are in millimeters)
    distance_meters = median_depth / 1000.0
    
    return distance_meters

def visualize_detections(color_image, depth_image, detections, distance_threshold):
    """
    Draw bounding boxes and distance information on the image.
    
    Args:
        color_image: RGB image
        depth_image: Depth image
        detections: List of person detections
        distance_threshold: Distance threshold in meters
        
    Returns:
        Annotated image
    """
    # Create a copy of the color image for visualization
    vis_image = color_image.copy()
    
    if not detections:
        return vis_image
    
    # Process each detection
    for detection in detections:
        # Get bounding box coordinates
        x1, y1, x2, y2 = detection['box']
        confidence = detection['confidence']
        
        # Calculate distance
        distance = calculate_distance(depth_image, x1, y1, x2, y2)
        
        # Determine color based on distance threshold
        if distance is not None and distance <= distance_threshold:
            color = (0, 255, 0)  # Green for people within threshold
            thickness = 3
        else:
            color = (0, 0, 255)  # Red for people beyond threshold
            thickness = 2
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        if distance is not None:
            label = f"Person: {confidence:.2f}, {distance:.2f}m"
        else:
            label = f"Person: {confidence:.2f}, No depth"
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(vis_image, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(vis_image, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return vis_image

def simulate_camera(net, distance_threshold):
    """Simulate camera operation with person detection when no physical camera is available."""
    print("\n" + "="*50)
    print("SIMULATION MODE WITH PERSON DETECTION")
    print("No physical camera detected - running in simulation mode")
    print("="*50 + "\n")
    
    # Create simulated color and depth images with matching height
    height = 720
    width = 1280
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    depth_image = np.zeros((height, width), dtype=np.uint16)
    
    # Create a gradient pattern for the color image
    for i in range(color_image.shape[0]):
        for j in range(color_image.shape[1]):
            color_image[i, j] = [
                int(255 * i / color_image.shape[0]),
                int(255 * j / color_image.shape[1]),
                128
            ]
    
    # Create a pattern for the depth image (simulating depth)
    center_x, center_y = depth_image.shape[1] // 2, depth_image.shape[0] // 2
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            # Calculate distance from center
            distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            # Create a circular pattern
            depth_image[i, j] = int(10000 * (1 - min(1, distance / max(center_x, center_y))))
    
    # Add simulated people to the color image
    # Person 1 - close
    person1_x, person1_y = 300, 300
    person1_w, person1_h = 100, 200
    color_image[person1_y:person1_y+person1_h, person1_x:person1_x+person1_w] = [0, 0, 200]
    # Set depth for person 1 (1.5m)
    depth_image[person1_y:person1_y+person1_h, person1_x:person1_x+person1_w] = 1500
    
    # Person 2 - far
    person2_x, person2_y = 800, 400
    person2_w, person2_h = 80, 160
    color_image[person2_y:person2_y+person2_h, person2_x:person2_x+person2_w] = [0, 0, 200]
    # Set depth for person 2 (3.5m)
    depth_image[person2_y:person2_y+person2_h, person2_x:person2_x+person2_w] = 3500
    
    # Apply colormap on depth image
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    # Add text to images
    cv2.putText(color_image, "SIMULATED RGB STREAM", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(depth_colormap, "SIMULATED DEPTH STREAM", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Print simulated camera information
    print("\n" + "="*50)
    print("SIMULATED CAMERA INFORMATION")
    print("="*50)
    print("Name: Intel RealSense D457 (Simulated)")
    print("Serial Number: SIMULATED-D457-SN")
    print("Firmware Version: 5.15.1.0 (Simulated)")
    print("Product Line: D400 (Simulated)")
    print("="*50 + "\n")
    
    # Create simulated detections
    simulated_detections = [
        {
            'box': (person1_x, person1_y, person1_x + person1_w, person1_y + person1_h),
            'confidence': 0.95
        },
        {
            'box': (person2_x, person2_y, person2_x + person2_w, person2_y + person2_h),
            'confidence': 0.85
        }
    ]
    
    # Visualize detections
    vis_image = visualize_detections(color_image, depth_image, simulated_detections, distance_threshold)
    
    # Apply colormap on depth image
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    # Stack both images horizontally
    images = np.hstack((vis_image, depth_colormap))
    
    # Save images to files
    cv2.imwrite('simulated_rgb_with_detection.jpg', vis_image)
    cv2.imwrite('simulated_depth.jpg', depth_colormap)
    cv2.imwrite('simulated_combined_with_detection.jpg', images)
    
    print("\nSimulated frames with person detection generated successfully!")
    print("Images saved as:")
    print("- simulated_rgb_with_detection.jpg")
    print("- simulated_depth.jpg")
    print("- simulated_combined_with_detection.jpg")
    
    return True

def main():
    """Main function to initialize camera and perform person detection."""
    parser = argparse.ArgumentParser(description='Person Detection with Distance Calculation')
    parser.add_argument('--simulate', action='store_true', help='Run in simulation mode')
    parser.add_argument('--threshold', type=float, default=DEFAULT_DISTANCE_THRESHOLD, 
                        help=f'Distance threshold in meters (default: {DEFAULT_DISTANCE_THRESHOLD}m)')
    args = parser.parse_args()
    
    # Load person detection model
    print("Loading person detection model...")
    net = load_person_detection_model()
    if net is None and not args.simulate:
        print("Error: Failed to load person detection model.")
        return False
    
    if args.simulate:
        return simulate_camera(net, args.threshold)
    
    print("Initializing Intel RealSense D457 camera...")
    
    try:
        # Create a context object to manage RealSense devices
        ctx = rs.context()
        
        # Check if any devices are connected
        devices = ctx.query_devices()
        device_count = devices.size()
        
        if device_count == 0:
            print("Error: No RealSense devices detected. Please connect a camera and try again.")
            print("Tip: Run with --simulate flag to test in simulation mode.")
            return False
        
        print(f"Found {device_count} RealSense device(s)")
        
        # Get the first device
        device = devices[0]
        
        # Print camera information
        print_camera_info(device)
        
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable streams with D457 supported resolutions
        # RGB: 1280x720 @ 30fps
        # Depth: 1280x720 @ 30fps
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        # Start streaming
        print("Starting camera streams...")
        pipeline_profile = pipeline.start(config)
        
        # Get the selected device
        selected_device = pipeline_profile.get_device()
        print(f"Using device: {selected_device.get_info(rs.camera_info.name)}")
        
        # Wait for camera to stabilize
        print("Waiting for camera to stabilize...")
        time.sleep(2)
        
        # Create alignment object to align depth frames to color frames
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        # Process frames with person detection
        print("Starting person detection...")
        
        try:
            # Wait for a coherent pair of frames: depth and color
            for i in range(30):  # Skip first few frames to allow auto-exposure to stabilize
                frames = pipeline.wait_for_frames()
            
            # Get aligned frames
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                print("Error: Could not capture both depth and color frames.")
                return False
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Detect people in the color image
            detections = detect_people(net, color_image)
            
            # Visualize detections with distance information
            vis_image = visualize_detections(color_image, depth_image, detections, args.threshold)
            
            # Apply colormap on depth image
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Stack both images horizontally
            images = np.hstack((vis_image, depth_colormap))
            
            # Save images to files
            cv2.imwrite('camera_rgb_with_detection.jpg', vis_image)
            cv2.imwrite('camera_depth.jpg', depth_colormap)
            cv2.imwrite('camera_combined_with_detection.jpg', images)
            
            print("\nPerson detection completed successfully!")
            print("Images saved as:")
            print("- camera_rgb_with_detection.jpg")
            print("- camera_depth.jpg")
            print("- camera_combined_with_detection.jpg")
            
            # Optional: Run in continuous mode
            # Uncomment the following code to run in continuous mode
            """
            try:
                while True:
                    # Wait for a coherent pair of frames: depth and color
                    frames = pipeline.wait_for_frames()
                    
                    # Get aligned frames
                    aligned_frames = align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    
                    if not depth_frame or not color_frame:
                        continue
                    
                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Detect people in the color image
                    detections = detect_people(net, color_image)
                    
                    # Visualize detections with distance information
                    vis_image = visualize_detections(color_image, depth_image, detections, args.threshold)
                    
                    # Apply colormap on depth image
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    
                    # Stack both images horizontally
                    images = np.hstack((vis_image, depth_colormap))
                    
                    # Show images
                    cv2.namedWindow('Person Detection', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('Person Detection', images)
                    
                    # Exit on ESC key
                    if cv2.waitKey(1) == 27:
                        break
            
            except KeyboardInterrupt:
                print("Stopping...")
            
            finally:
                cv2.destroyAllWindows()
            """
            
        finally:
            # Stop streaming
            pipeline.stop()
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Run with --simulate flag to test in simulation mode.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("Person detection completed successfully.")
        sys.exit(0)
    else:
        print("Person detection failed.")
        sys.exit(1)