#!/usr/bin/env python3
"""
Intel RealSense D457 Camera Setup Script

This script initializes a connection to the Intel RealSense D457 depth camera,
captures and displays test frames from both RGB and depth streams,
and prints basic camera information.

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

def simulate_camera():
    """Simulate camera operation when no physical camera is available."""
    print("\n" + "="*50)
    print("SIMULATION MODE")
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
    
    # Stack both images horizontally
    images = np.hstack((color_image, depth_colormap))
    
    # Save images to files
    cv2.imwrite('simulated_rgb.jpg', color_image)
    cv2.imwrite('simulated_depth.jpg', depth_colormap)
    cv2.imwrite('simulated_combined.jpg', images)
    
    print("\nSimulated frames generated successfully!")
    print("Images saved as:")
    print("- simulated_rgb.jpg")
    print("- simulated_depth.jpg")
    print("- simulated_combined.jpg")
    
    return True

def main():
    """Main function to initialize and test the RealSense camera."""
    parser = argparse.ArgumentParser(description='Intel RealSense D457 Camera Setup')
    parser.add_argument('--simulate', action='store_true', help='Run in simulation mode')
    args = parser.parse_args()
    
    if args.simulate:
        return simulate_camera()
    
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
        # RGB: 1280x720 @ 30fps (changed from 800p to match depth)
        # Depth: 1280x720 @ 90fps (max)
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
        
        # Capture and display frames
        print("Capturing test frames...")
        
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
            
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            
            # Save images to files
            cv2.imwrite('camera_rgb.jpg', color_image)
            cv2.imwrite('camera_depth.jpg', depth_colormap)
            cv2.imwrite('camera_combined.jpg', images)
            
            print("\nTest frames captured successfully!")
            print("Images saved as:")
            print("- camera_rgb.jpg")
            print("- camera_depth.jpg")
            print("- camera_combined.jpg")
            
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
        print("Camera setup completed successfully.")
        sys.exit(0)
    else:
        print("Camera setup failed.")
        sys.exit(1)