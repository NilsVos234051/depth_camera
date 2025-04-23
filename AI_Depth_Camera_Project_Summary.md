# AI Depth Camera Project Summary

## Project Overview
This document summarizes the key information extracted from the ProjectBrief_AIDepthCamera.pdf file and supplementary research on the Intel RealSense D457 depth camera.

## Project Brief Summary
The project involves creating a creative and visually engaging demo that leverages depth and RGBD data using the Intel RealSense D457 camera. The final demo will be displayed during an Open Day event to attract passersby with real-time visualizations.

### Key Project Requirements
1. **Creativity**
   - Unique and visually engaging idea
   - Ability to spark interest and stand out to viewers

2. **Functionality with Dataset / Demo**
   - Concept works with online dataset or simulation
   - Demonstrated running effectively in own setup

3. **Successful RealSense Integration**
   - Final version running on the RealSense camera
   - Clear demonstration during live Open Day demo

### Deliverables
- Project Code / Notebook (Jupyter Notebook, Python script, or equivalent)
- Short Demo Video (approximately 15 seconds)
- Live Camera Demo during assigned time slot

## Intel RealSense D457 Camera Specifications

### Technical Specifications
- **Resolution**
  - Video Resolution: 1280 x 720 @ 90 fps
  - RGB Frame Resolution: 1280 × 800 @ 30 fps
  - Depth Resolution: Up to 1280x720

- **Field of View**
  - Horizontal: 87° (±3°)
  - Vertical: 58° (±3°)
  - Diagonal: 89-98 degrees

- **Depth Technology**
  - Type: Stereoscopic depth sensing
  - Operational Range: 0.6m - 6m
  - Depth Accuracy: <2% at 4m
  - Minimum Depth Distance: ~52 cm

- **Connectivity and Interfaces**
  - Connectivity Type: Wired
  - Interfaces: USB-C, FAKRA, Ethernet
  - System Interface: GMSL/FAKRA

- **Additional Features**
  - RGB Sensor: Global Shutter
  - Tracking Module
  - Inertial Measurement Unit (IMU)
  - All-Pass depth filter
  - IP65 compliant (protected from dust and water)
  - Operating Temperature: 32°F - 95°F

- **Physical Specifications**
  - Dimensions: 124 mm × 29 mm × 36 mm
  - Weight: Approximately 4.94 oz

## API and SDK Information

### Intel RealSense SDK 2.0
- Open-source, cross-platform, OS-independent (Windows, Linux, Android)
- Available on GitHub: IntelRealSense/librealsense

### Programming Language Support
- C++ (Doxygen documentation)
- Python (Sphinx documentation)
- ROS2 (Robot Operating System)

### Key API Features
- Depth and color streaming
- Intrinsic and extrinsic calibration information
- Synthetic streams
- Configuration of depth cameras
- Control of camera settings
- Access to streaming data

### Python API Usage
The camera is accessed through the `pyrealsense2` library:

```python
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Start streaming
pipeline.start(config)
```

## Resources
- Official Documentation: https://www.intelrealsense.com/depth-camera-d457/
- Developer Documentation: https://dev.intelrealsense.com/docs/
- GitHub Repository: https://github.com/IntelRealSense/librealsense

## Project Goal
Create an innovative, fun, and visually impressive application using the Intel RealSense D457 camera that will make people stop and say, "That's awesome!" Potential ideas include gesture tracking, 3D scene generation, AI-driven artwork, or real-time object interaction.