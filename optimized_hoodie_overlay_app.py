#!/usr/bin/env python3
"""
Optimized Hoodie Overlay Application

This application integrates person detection with 3D model overlay to create a unified system that:
1. Detects people using the Intel RealSense D457 camera
2. Calculates their distance from the camera
3. Overlays a 3D hoodie model on detected people within a specified distance threshold
4. Adjusts the hoodie model's size, position, and orientation based on the person's position and size

The application works in both real camera and simulation modes.

This version includes performance optimizations for real-time processing:
- Adaptive frame skipping
- Resolution adjustment
- Improved threading model
- Enhanced 3D rendering integration
- Comprehensive error handling and logging
"""

import os
import sys
import time
import argparse
import threading
import queue
import logging
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from collections import deque

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
DEFAULT_CONFIG_FILE = "hoodie_overlay_config.json"
DEFAULT_LOG_LEVEL = "INFO"

# Performance settings
DEFAULT_FRAME_SKIP = 0  # 0 means no frame skipping
DEFAULT_DOWNSCALE_FACTOR = 1.0  # 1.0 means no downscaling
DEFAULT_DETECTION_INTERVAL = 1  # Process every Nth frame for detection

class PerformanceMonitor:
    """Class to monitor and adjust performance parameters."""
    
    def __init__(self, target_fps=25.0, window_size=30):
        """
        Initialize the performance monitor.
        
        Args:
            target_fps: Target frames per second
            window_size: Number of frames to consider for FPS calculation
        """
        self.target_fps = target_fps
        self.frame_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)
        self.rendering_times = deque(maxlen=window_size)
        self.last_frame_time = None
        self.downscale_factor = DEFAULT_DOWNSCALE_FACTOR
        self.frame_skip = DEFAULT_FRAME_SKIP
        self.detection_interval = DEFAULT_DETECTION_INTERVAL
        self.stats = {
            'fps': 0.0,
            'avg_processing_time': 0.0,
            'avg_detection_time': 0.0,
            'avg_rendering_time': 0.0,
            'dropped_frames': 0,
            'processed_frames': 0,
            'detection_interval': self.detection_interval,
            'downscale_factor': self.downscale_factor
        }
    
    def start_frame(self):
        """Mark the start of a frame."""
        self.last_frame_time = time.time()
    
    def end_frame(self):
        """Mark the end of a frame and update statistics."""
        if self.last_frame_time is not None:
            frame_time = time.time() - self.last_frame_time
            self.frame_times.append(frame_time)
            self.update_stats()
    
    def record_processing_time(self, duration):
        """Record the time taken for frame processing."""
        self.processing_times.append(duration)
    
    def record_detection_time(self, duration):
        """Record the time taken for person detection."""
        self.detection_times.append(duration)
    
    def record_rendering_time(self, duration):
        """Record the time taken for 3D rendering."""
        self.rendering_times.append(duration)
    
    def update_stats(self):
        """Update performance statistics."""
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.stats['fps'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        if len(self.processing_times) > 0:
            self.stats['avg_processing_time'] = sum(self.processing_times) / len(self.processing_times)
        
        if len(self.detection_times) > 0:
            self.stats['avg_detection_time'] = sum(self.detection_times) / len(self.detection_times)
        
        if len(self.rendering_times) > 0:
            self.stats['avg_rendering_time'] = sum(self.rendering_times) / len(self.rendering_times)
        
        self.stats['detection_interval'] = self.detection_interval
        self.stats['downscale_factor'] = self.downscale_factor
    
    def adjust_parameters(self):
        """
        Adjust performance parameters based on current FPS.
        Returns True if parameters were changed.
        """
        if len(self.frame_times) < self.frame_times.maxlen // 2:
            # Not enough data yet
            return False
        
        current_fps = self.stats['fps']
        changed = False
        
        # If FPS is too low, try to improve performance
        if current_fps < self.target_fps * 0.8:
            # First try increasing detection interval
            if self.detection_interval < 3:
                self.detection_interval += 1
                changed = True
                logging.info(f"Increasing detection interval to {self.detection_interval}")
            # Then try increasing downscale factor
            elif self.downscale_factor < 2.0:
                self.downscale_factor += 0.25
                changed = True
                logging.info(f"Increasing downscale factor to {self.downscale_factor}")
            # Finally try frame skipping
            elif self.frame_skip < 2:
                self.frame_skip += 1
                changed = True
                logging.info(f"Increasing frame skip to {self.frame_skip}")
        
        # If FPS is comfortably above target, try to improve quality
        elif current_fps > self.target_fps * 1.2:
            # First reduce frame skipping
            if self.frame_skip > 0:
                self.frame_skip -= 1
                changed = True
                logging.info(f"Decreasing frame skip to {self.frame_skip}")
            # Then reduce downscale factor
            elif self.downscale_factor > 1.0:
                self.downscale_factor -= 0.25
                changed = True
                logging.info(f"Decreasing downscale factor to {self.downscale_factor}")
            # Finally reduce detection interval
            elif self.detection_interval > 1:
                self.detection_interval -= 1
                changed = True
                logging.info(f"Decreasing detection interval to {self.detection_interval}")
        
        return changed
    
    def get_stats(self):
        """Get current performance statistics."""
        return self.stats
    
    def should_skip_frame(self, frame_count):
        """Determine if the current frame should be skipped."""
        return self.frame_skip > 0 and frame_count % (self.frame_skip + 1) != 0
    
    def should_detect(self, frame_count):
        """Determine if person detection should be performed on this frame."""
        return frame_count % self.detection_interval == 0


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
                - log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                - config_file: Path to configuration file
                - target_fps: Target frames per second
                - frame_skip: Number of frames to skip
                - downscale_factor: Factor to downscale images for processing
                - detection_interval: Process every Nth frame for detection
        """
        self.config = config
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)  # Queue for frames to be processed
        self.result_queue = queue.Queue(maxsize=5)  # Queue for processed results
        
        # Set up logging
        self.setup_logging()
        
        # Load configuration from file if specified
        self.load_config()
        
        # Initialize performance monitor
        self.perf_monitor = PerformanceMonitor(
            target_fps=self.config.get('target_fps', 25.0)
        )
        
        # Initialize person detection model
        logging.info("Loading person detection model...")
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
        
        # Person tracking data
        self.previous_detections = []
        
        # Create output directory if it doesn't exist
        if self.config.get('output_dir'):
            os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Frame counter
        self.frame_count = 0
        
        # Last performance adjustment time
        self.last_adjustment_time = time.time()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_level_str = self.config.get('log_level', DEFAULT_LOG_LEVEL)
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('hoodie_overlay.log', mode='w')
            ]
        )
        
        logging.info("Logging initialized")
    
    def load_config(self):
        """Load configuration from file if specified."""
        config_file = self.config.get('config_file', DEFAULT_CONFIG_FILE)
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Update config with file values, but don't overwrite command-line args
                for key, value in file_config.items():
                    if key not in self.config or self.config[key] is None:
                        self.config[key] = value
                
                logging.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logging.error(f"Error loading configuration file: {e}")
        else:
            logging.info(f"Configuration file {config_file} not found, using defaults")
    
    def save_config(self):
        """Save current configuration to file."""
        config_file = self.config.get('config_file', DEFAULT_CONFIG_FILE)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            logging.info(f"Saved configuration to {config_file}")
        except Exception as e:
            logging.error(f"Error saving configuration file: {e}")
    
    def setup_panda3d(self):
        """Set up Panda3D for 3D rendering without opening a window."""
        try:
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
            
            logging.info("Panda3D rendering setup complete")
        except Exception as e:
            logging.error(f"Error setting up Panda3D: {e}")
            raise
    
    def initialize_camera(self):
        """Initialize the RealSense camera or set up simulation."""
        if self.config['simulate']:
            logging.info("Running in simulation mode")
            return True
        
        try:
            # Create a context object to manage RealSense devices
            ctx = rs.context()
            
            # Check if any devices are connected
            devices = ctx.query_devices()
            device_count = devices.size()
            
            if device_count == 0:
                logging.error("No RealSense devices detected. Please connect a camera and try again.")
                logging.info("Tip: Run with --simulate flag to test in simulation mode.")
                return False
            
            logging.info(f"Found {device_count} RealSense device(s)")
            
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
            logging.info("Starting camera streams...")
            pipeline_profile = self.pipeline.start(config)
            
            # Get the selected device
            selected_device = pipeline_profile.get_device()
            logging.info(f"Using device: {selected_device.get_info(rs.camera_info.name)}")
            
            # Wait for camera to stabilize
            logging.info("Waiting for camera to stabilize...")
            time.sleep(2)
            
            # Create alignment object to align depth frames to color frames
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            return True
            
        except Exception as e:
            logging.error(f"Error initializing camera: {e}")
            logging.info("Tip: Run with --simulate flag to test in simulation mode.")
            return False
    
    def capture_frames(self):
        """Capture frames from the camera or generate simulated frames."""
        logging.info("Starting frame capture thread")
        
        if self.config['simulate']:
            # Generate simulated frames
            while self.running:
                try:
                    self.perf_monitor.start_frame()
                    
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
                    
                    # Apply frame skipping if needed
                    if not self.perf_monitor.should_skip_frame(self.frame_count):
                        # Put frames in the queue
                        try:
                            self.frame_queue.put((self.frame_count, color_image, depth_image), block=False)
                        except queue.Full:
                            # Skip frame if queue is full
                            logging.debug("Frame queue full, skipping frame")
                    
                    self.frame_count += 1
                    self.perf_monitor.end_frame()
                    
                    # Simulate frame rate
                    time.sleep(1.0 / self.config['camera_fps'])
                    
                    # Periodically adjust performance parameters
                    if time.time() - self.last_adjustment_time > 5.0:
                        if self.perf_monitor.adjust_parameters():
                            self.last_adjustment_time = time.time()
                
                except Exception as e:
                    logging.error(f"Error in capture thread (simulation): {e}")
        else:
            # Capture frames from the RealSense camera
            try:
                while self.running:
                    try:
                        self.perf_monitor.start_frame()
                        
                        # Wait for a coherent pair of frames: depth and color
                        frames = self.pipeline.wait_for_frames()
                        
                        # Get aligned frames
                        aligned_frames = self.align.process(frames)
                        depth_frame = aligned_frames.get_depth_frame()
                        color_frame = aligned_frames.get_color_frame()
                        
                        if not depth_frame or not color_frame:
                            logging.warning("Missing depth or color frame, skipping")
                            continue
                        
                        # Convert images to numpy arrays
                        depth_image = np.asanyarray(depth_frame.get_data())
                        color_image = np.asanyarray(color_frame.get_data())
                        
                        # Apply frame skipping if needed
                        if not self.perf_monitor.should_skip_frame(self.frame_count):
                            # Put frames in the queue
                            try:
                                self.frame_queue.put((self.frame_count, color_image, depth_image), block=False)
                            except queue.Full:
                                # Skip frame if queue is full
                                logging.debug("Frame queue full, skipping frame")
                        
                        self.frame_count += 1
                        self.perf_monitor.end_frame()
                        
                        # Periodically adjust performance parameters
                        if time.time() - self.last_adjustment_time > 5.0:
                            if self.perf_monitor.adjust_parameters():
                                self.last_adjustment_time = time.time()
                    
                    except Exception as e:
                        logging.error(f"Error processing frame: {e}")
            
            except Exception as e:
                logging.error(f"Error in capture thread: {e}")
                self.running = False
    
    def process_frames(self):
        """Process frames to detect people and overlay hoodie models."""
        logging.info("Starting frame processing thread")
        
        while self.running:
            try:
                # Get frames from the queue
                frame_count, color_image, depth_image = self.frame_queue.get(timeout=1.0)
                
                start_time = time.time()
                
                # Apply downscaling for processing if needed
                downscale_factor = self.perf_monitor.downscale_factor
                if downscale_factor > 1.0:
                    proc_width = int(color_image.shape[1] / downscale_factor)
                    proc_height = int(color_image.shape[0] / downscale_factor)
                    proc_image = cv2.resize(color_image, (proc_width, proc_height))
                else:
                    proc_image = color_image
                
                # Detect people in the color image (only on selected frames)
                detections = []
                if self.perf_monitor.should_detect(frame_count):
                    detection_start = time.time()
                    detections = detect_people(self.detection_model, proc_image)
                    detection_time = time.time() - detection_start
                    self.perf_monitor.record_detection_time(detection_time)
                    
                    # If we downscaled, adjust detection coordinates
                    if downscale_factor > 1.0:
                        for detection in detections:
                            x1, y1, x2, y2 = detection['box']
                            detection['box'] = (
                                int(x1 * downscale_factor),
                                int(y1 * downscale_factor),
                                int(x2 * downscale_factor),
                                int(y2 * downscale_factor)
                            )
                    
                    # Update tracking
                    self.previous_detections = detections
                else:
                    # Use previous detections for tracking continuity
                    detections = self.previous_detections
                
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
                
                # Render the 3D models
                rendering_start = time.time()
                # Create a visualization image
                vis_image = self.visualize_detections(color_image, depth_image, processed_detections)
                rendering_time = time.time() - rendering_start
                self.perf_monitor.record_rendering_time(rendering_time)
                
                # Apply colormap on depth image
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Add performance stats to the image
                stats = self.perf_monitor.get_stats()
                self.add_stats_overlay(vis_image, stats)
                
                # Stack both images horizontally
                combined_image = np.hstack((vis_image, depth_colormap))
                
                # Record total processing time
                processing_time = time.time() - start_time
                self.perf_monitor.record_processing_time(processing_time)
                
                # Put the result in the queue
                try:
                    self.result_queue.put((frame_count, vis_image, depth_colormap, combined_image), block=False)
                except queue.Full:
                    # Skip result if queue is full
                    logging.debug("Result queue full, skipping result")
                
                # Mark the frame as processed
                self.frame_queue.task_done()
                
            except queue.Empty:
                # No frames available, continue
                continue
            except Exception as e:
                logging.error(f"Error processing frames: {e}")
    
    def add_stats_overlay(self, image, stats):
        """Add performance statistics overlay to the image."""
        # Prepare text lines
        lines = [
            f"FPS: {stats['fps']:.1f}",
            f"Processing: {stats['avg_processing_time']*1000:.1f}ms",
            f"Detection: {stats['avg_detection_time']*1000:.1f}ms",
            f"Rendering: {stats['avg_rendering_time']*1000:.1f}ms",
            f"Scale: {stats['downscale_factor']:.2f}x",
            f"Skip: {self.perf_monitor.frame_skip}",
            f"Det.Int: {stats['detection_interval']}"
        ]
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (200, 30 + 20 * len(lines)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # Draw text
        for i, line in enumerate(lines):
            cv2.putText(image, line, (15, 30 + 20 * i), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
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
        logging.info("Starting result saving thread")
        
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
                    logging.info(f"Processed frame {frame_count}")
                
                # Mark the result as processed
                self.result_queue.task_done()
                
                # Stop after processing a few frames in simulation mode
                if self.config['simulate'] and frame_count >= 5:
                    logging.info(f"Processed {frame_count + 1} frames in simulation mode. Stopping.")
                    self.running = False
                
            except queue.Empty:
                # No results available, continue
                continue
            except Exception as e:
                logging.error(f"Error saving results: {e}")
    
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
            logging.info("Stopping...")
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
            
            # Save final configuration
            self.save_config()
        
        return True


def main():
    """Main function to parse arguments and run the application."""
    parser = argparse.ArgumentParser(description='Optimized Hoodie Overlay Application')
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
    parser.add_argument('--log-level', type=str, default=DEFAULT_LOG_LEVEL,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help=f'Logging level (default: {DEFAULT_LOG_LEVEL})')
    parser.add_argument('--config-file', type=str, default=DEFAULT_CONFIG_FILE,
                        help=f'Configuration file path (default: {DEFAULT_CONFIG_FILE})')
    parser.add_argument('--target-fps', type=float, default=25.0,
                        help='Target frames per second (default: 25.0)')
    parser.add_argument('--frame-skip', type=int, default=DEFAULT_FRAME_SKIP,
                        help=f'Number of frames to skip (default: {DEFAULT_FRAME_SKIP})')
    parser.add_argument('--downscale', type=float, default=DEFAULT_DOWNSCALE_FACTOR,
                        help=f'Factor to downscale images for processing (default: {DEFAULT_DOWNSCALE_FACTOR})')
    parser.add_argument('--detection-interval', type=int, default=DEFAULT_DETECTION_INTERVAL,
                        help=f'Process every Nth frame for detection (default: {DEFAULT_DETECTION_INTERVAL})')
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
        'output_dir': args.output_dir,
        'log_level': args.log_level,
        'config_file': args.config_file,
        'target_fps': args.target_fps,
        'frame_skip': args.frame_skip,
        'downscale_factor': args.downscale,
        'detection_interval': args.detection_interval
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