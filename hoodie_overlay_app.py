#!/usr/bin/env python3
"""
Hoodie Overlay Application

This application integrates person detection with 3D model overlay to create a unified system that:
1. Detects people using the Intel RealSense D457 camera
2. Calculates their distance from the camera
3. Overlays a 3D hoodie model on detected people within a specified distance threshold
4. Adjusts the hoodie model's size, position, and orientation based on the person's position and size

The application works in both real camera and simulation modes.

Requirements:
- pyrealsense2
- numpy
- opencv-python
- panda3d
"""

import os
import sys
import time
import argparse
import threading
import queue
import numpy as np
import cv2
import pyrealsense2 as rs

# Import person detection functionality
from person_detection import (
    load_person_detection_model,
    detect_people,
    calculate_distance,
    print_camera_info,
    simulate_camera
)

# Import hoodie model functionality
from hoodie_model import HoodieModel

# Default configuration values
DEFAULT_DISTANCE_THRESHOLD = 2.0  # meters
DEFAULT_HOODIE_SCALE = 1.0
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_FPS = 30

class HoodieOverlayApp:
    """Main application class that integrates person detection with 3D hoodie model overlay."""
    
    def __init__(self, config):
        """
        Initialize the application with the given configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                - simulate: Boolean indicating whether to run in simulation mode
                - distance_threshold: Distance threshold in meters for hoodie overlay
                - hoodie_scale: Scale factor for the hoodie model
                - model_path: Path to the 3D model file (optional)
                - camera_width: Camera width in pixels
                - camera_height: Camera height in pixels
                - camera_fps: Camera frames per second
                - headless: Boolean indicating whether to run in headless mode
                - output_dir: Directory to save output files
        """
        self.config = config
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)  # Queue for frames to be processed
        self.result_queue = queue.Queue(maxsize=5)  # Queue for processed results
        
        # Initialize person detection model
        print("Loading person detection model...")
        self.detection_model = load_person_detection_model()
        
        # Initialize camera pipeline
        self.pipeline = None
        self.align = None
        
        # Initialize Panda3D for 3D rendering
        self.setup_panda3d()
        
        # Initialize hoodie model
        self.hoodie_model = HoodieModel(config.get('model_path'))
        self.hoodie_model.load_model(self.render)
        
        # Create a mapping of person IDs to hoodie models
        # (for tracking multiple people and assigning models)
        self.person_hoodies = {}
        self.next_person_id = 0
        
        # Create output directory if it doesn't exist
        if self.config.get('output_dir'):
            os.makedirs(self.config['output_dir'], exist_ok=True)
        
    def setup_panda3d(self):
        """Set up Panda3D for 3D rendering without opening a window."""
        # Import Panda3D modules here to avoid circular imports
        from panda3d.core import loadPrcFileData, NodePath, PandaNode
        from panda3d.core import AmbientLight, DirectionalLight, Vec4
        
        # Configure Panda3D to run without a window
        loadPrcFileData("", "window-type none")
        loadPrcFileData("", "audio-library-name null")
        
        # Create a dummy render node
        self.render = NodePath(PandaNode("render"))
        
        # Set up basic lighting
        # Ambient light
        ambient_light = AmbientLight("ambient_light")
        ambient_light.setColor(Vec4(0.3, 0.3, 0.3, 1))
        ambient_node = self.render.attachNewNode(ambient_light)
        self.render.setLight(ambient_node)
        
        # Directional light (key light)
        key_light = DirectionalLight("key_light")
        key_light.setColor(Vec4(0.8, 0.8, 0.8, 1))
        key_node = self.render.attachNewNode(key_light)
        key_node.setHpr(45, -45, 0)
        self.render.setLight(key_node)
    
    def initialize_camera(self):
        """Initialize the RealSense camera or set up simulation."""
        if self.config['simulate']:
            print("Running in simulation mode")
            return True
        
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
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable streams with configured resolution and frame rate
            config.enable_stream(
                rs.stream.depth, 
                self.config['camera_width'], 
                self.config['camera_height'], 
                rs.format.z16, 
                self.config['camera_fps']
            )
            config.enable_stream(
                rs.stream.color, 
                self.config['camera_width'], 
                self.config['camera_height'], 
                rs.format.bgr8, 
                self.config['camera_fps']
            )
            
            # Start streaming
            print("Starting camera streams...")
            pipeline_profile = self.pipeline.start(config)
            
            # Get the selected device
            selected_device = pipeline_profile.get_device()
            print(f"Using device: {selected_device.get_info(rs.camera_info.name)}")
            
            # Wait for camera to stabilize
            print("Waiting for camera to stabilize...")
            time.sleep(2)
            
            # Create alignment object to align depth frames to color frames
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            print("Tip: Run with --simulate flag to test in simulation mode.")
            return False
    
    def capture_frames(self):
        """Capture frames from the camera or generate simulated frames."""
        if self.config['simulate']:
            # Generate simulated frames
            while self.running:
                # Create simulated color and depth images
                height = self.config['camera_height']
                width = self.config['camera_width']
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
                
                # Put frames in the queue
                try:
                    self.frame_queue.put((color_image, depth_image), block=False)
                except queue.Full:
                    # Skip frame if queue is full
                    pass
                
                # Simulate frame rate
                time.sleep(1.0 / self.config['camera_fps'])
        else:
            # Capture frames from the RealSense camera
            try:
                while self.running:
                    # Wait for a coherent pair of frames: depth and color
                    frames = self.pipeline.wait_for_frames()
                    
                    # Get aligned frames
                    aligned_frames = self.align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    
                    if not depth_frame or not color_frame:
                        continue
                    
                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Put frames in the queue
                    try:
                        self.frame_queue.put((color_image, depth_image), block=False)
                    except queue.Full:
                        # Skip frame if queue is full
                        pass
            except Exception as e:
                print(f"Error capturing frames: {e}")
                self.running = False
    
    def process_frames(self):
        """Process frames to detect people and overlay hoodie models."""
        frame_count = 0
        while self.running:
            try:
                # Get frames from the queue
                color_image, depth_image = self.frame_queue.get(timeout=1.0)
                
                # Detect people in the color image
                detections = detect_people(self.detection_model, color_image)
                
                # Process each detection
                processed_detections = []
                for detection in detections:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = detection['box']
                    confidence = detection['confidence']
                    
                    # Calculate distance
                    distance = calculate_distance(depth_image, x1, y1, x2, y2)
                    
                    # Calculate person dimensions
                    person_width = x2 - x1
                    person_height = y2 - y1
                    
                    # Calculate person center position
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Add to processed detections
                    processed_detections.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'distance': distance,
                        'width': person_width,
                        'height': person_height,
                        'center': (center_x, center_y)
                    })
                
                # Create a visualization image
                vis_image = self.visualize_detections(color_image, depth_image, processed_detections)
                
                # Apply colormap on depth image
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Stack both images horizontally
                combined_image = np.hstack((vis_image, depth_colormap))
                
                # Put the result in the queue
                try:
                    self.result_queue.put((frame_count, vis_image, depth_colormap, combined_image), block=False)
                except queue.Full:
                    # Skip result if queue is full
                    pass
                
                # Mark the frame as processed
                self.frame_queue.task_done()
                frame_count += 1
                
            except queue.Empty:
                # No frames available, continue
                continue
            except Exception as e:
                print(f"Error processing frames: {e}")
    
    def visualize_detections(self, color_image, depth_image, detections):
        """
        Visualize detections with hoodie overlay.
        
        Args:
            color_image: RGB image
            depth_image: Depth image
            detections: List of processed person detections
            
        Returns:
            Annotated image with hoodie overlays
        """
        # Create a copy of the color image for visualization
        vis_image = color_image.copy()
        
        if not detections:
            return vis_image
        
        # Process each detection
        for detection in detections:
            # Get detection information
            x1, y1, x2, y2 = detection['box']
            confidence = detection['confidence']
            distance = detection['distance']
            person_width = detection['width']
            person_height = detection['height']
            center_x, center_y = detection['center']
            
            # Determine color based on distance threshold
            if distance is not None and distance <= self.config['distance_threshold']:
                color = (0, 255, 0)  # Green for people within threshold
                thickness = 3
                
                # Calculate person dimensions in 3D space
                # Assuming a standard human height of 1.7 meters
                person_height_meters = 1.7
                
                # Calculate scale factor based on person's height in the image
                # This is a simplified calculation and might need adjustment
                scale_factor = person_height_meters / person_height * self.config['hoodie_scale']
                
                # Calculate 3D position
                # Convert from image coordinates to 3D space
                # This is a simplified conversion and might need adjustment
                x_3d = (center_x - color_image.shape[1] / 2) / color_image.shape[1] * distance
                y_3d = distance
                z_3d = (color_image.shape[0] / 2 - center_y) / color_image.shape[0] * distance
                
                # Adjust the hoodie model
                self.hoodie_model.set_scale(scale_factor)
                self.hoodie_model.set_position(x_3d, y_3d, z_3d)
                
                # Calculate orientation based on person's position
                # This is a simplified calculation and might need adjustment
                orientation = 0  # Default orientation
                self.hoodie_model.set_rotation(orientation, 0, 0)
                
                # Overlay the hoodie model on the person
                # This would require rendering the 3D model and compositing it with the image
                # For now, we'll just draw a placeholder
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(vis_image, "Hoodie Overlay", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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
    
    def save_results(self):
        """Save the processed results to files."""
        while self.running:
            try:
                # Get the result from the queue
                frame_count, vis_image, depth_colormap, combined_image = self.result_queue.get(timeout=1.0)
                
                # Save the images
                if self.config.get('output_dir'):
                    output_dir = self.config['output_dir']
                    cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count:04d}_rgb.jpg'), vis_image)
                    cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count:04d}_depth.jpg'), depth_colormap)
                    cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count:04d}_combined.jpg'), combined_image)
                else:
                    # Save to current directory
                    cv2.imwrite(f'frame_{frame_count:04d}_rgb.jpg', vis_image)
                    cv2.imwrite(f'frame_{frame_count:04d}_depth.jpg', depth_colormap)
                    cv2.imwrite(f'frame_{frame_count:04d}_combined.jpg', combined_image)
                
                # Print progress
                if frame_count % 10 == 0:
                    print(f"Processed frame {frame_count}")
                
                # Mark the result as processed
                self.result_queue.task_done()
                
                # Stop after processing a few frames in simulation mode
                if self.config['simulate'] and frame_count >= 5:
                    print(f"Processed {frame_count + 1} frames in simulation mode. Stopping.")
                    self.running = False
                
            except queue.Empty:
                # No results available, continue
                continue
            except Exception as e:
                print(f"Error saving results: {e}")
    
    def run(self):
        """Run the application."""
        # Initialize the camera
        if not self.initialize_camera():
            return False
        
        # Start the application
        self.running = True
        
        # Create threads for capturing, processing, and saving frames
        capture_thread = threading.Thread(target=self.capture_frames)
        process_thread = threading.Thread(target=self.process_frames)
        save_thread = threading.Thread(target=self.save_results)
        
        # Start the threads
        capture_thread.start()
        process_thread.start()
        save_thread.start()
        
        try:
            # Wait for threads to finish
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            # Stop the application
            self.running = False
            
            # Wait for threads to finish
            capture_thread.join()
            process_thread.join()
            save_thread.join()
            
            # Clean up
            if self.pipeline:
                self.pipeline.stop()
            
            # Clean up hoodie model
            self.hoodie_model.cleanup()
        
        return True


def main():
    """Main function to parse arguments and run the application."""
    parser = argparse.ArgumentParser(description='Hoodie Overlay Application')
    parser.add_argument('--simulate', action='store_true', help='Run in simulation mode')
    parser.add_argument('--threshold', type=float, default=DEFAULT_DISTANCE_THRESHOLD, 
                        help=f'Distance threshold in meters (default: {DEFAULT_DISTANCE_THRESHOLD}m)')
    parser.add_argument('--scale', type=float, default=DEFAULT_HOODIE_SCALE,
                        help=f'Hoodie model scale factor (default: {DEFAULT_HOODIE_SCALE})')
    parser.add_argument('--model', type=str, help='Path to the 3D model file')
    parser.add_argument('--width', type=int, default=DEFAULT_CAMERA_WIDTH,
                        help=f'Camera width in pixels (default: {DEFAULT_CAMERA_WIDTH})')
    parser.add_argument('--height', type=int, default=DEFAULT_CAMERA_HEIGHT,
                        help=f'Camera height in pixels (default: {DEFAULT_CAMERA_HEIGHT})')
    parser.add_argument('--fps', type=int, default=DEFAULT_CAMERA_FPS,
                        help=f'Camera frames per second (default: {DEFAULT_CAMERA_FPS})')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no display)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output files (default: output)')
    args = parser.parse_args()
    
    # Create configuration dictionary
    config = {
        'simulate': args.simulate,
        'distance_threshold': args.threshold,
        'hoodie_scale': args.scale,
        'model_path': args.model,
        'camera_width': args.width,
        'camera_height': args.height,
        'camera_fps': args.fps,
        'headless': args.headless or True,  # Always run in headless mode for now
        'output_dir': args.output_dir
    }
    
    # Create and run the application
    app = HoodieOverlayApp(config)
    success = app.run()
    
    if success:
        print("Hoodie overlay application completed successfully.")
        return 0
    else:
        print("Hoodie overlay application failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())