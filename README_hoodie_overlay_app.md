# Hoodie Overlay Application

This application integrates person detection with 3D model overlay to create a unified system that detects people using the Intel RealSense D457 camera and overlays a 3D hoodie model on them.

## Features

- Person detection using OpenCV DNN with MobileNet SSD
- Distance calculation using depth data from the RealSense camera
- 3D hoodie model overlay on detected people within a specified distance threshold
- Automatic adjustment of the hoodie model's size, position, and orientation based on the person's position and size
- Support for both real camera and simulation modes
- Configurable distance threshold, hoodie model scaling, and camera parameters

## Requirements

- Python 3.6+
- pyrealsense2
- numpy
- opencv-python
- panda3d

## Usage

```bash
python hoodie_overlay_app.py [options]
```

### Command Line Options

- `--simulate`: Run in simulation mode (no physical camera required)
- `--threshold THRESHOLD`: Distance threshold in meters (default: 2.0m)
- `--scale SCALE`: Hoodie model scale factor (default: 1.0)
- `--model MODEL`: Path to the 3D model file (optional)
- `--width WIDTH`: Camera width in pixels (default: 1280)
- `--height HEIGHT`: Camera height in pixels (default: 720)
- `--fps FPS`: Camera frames per second (default: 30)
- `--output-dir OUTPUT_DIR`: Directory to save output files (default: output)

## Examples

### Run in simulation mode with default settings

```bash
python hoodie_overlay_app.py --simulate
```

### Run with a custom distance threshold and scale

```bash
python hoodie_overlay_app.py --threshold 1.5 --scale 1.2
```

### Run with a custom 3D model file

```bash
python hoodie_overlay_app.py --model path/to/hoodie_model.obj
```

## Output

The application generates the following output files in the specified output directory:

- `frame_XXXX_rgb.jpg`: RGB image with person detection and hoodie overlay
- `frame_XXXX_depth.jpg`: Depth image with colormap applied
- `frame_XXXX_combined.jpg`: Combined RGB and depth images

## Components

The application integrates the following components:

1. **Person Detection**: Uses the MobileNet SSD model to detect people in the RGB image and calculates their distance using the depth data.

2. **3D Hoodie Model**: Loads and renders a 3D hoodie model using Panda3D, with support for scaling, positioning, and rotation.

3. **Camera Interface**: Interfaces with the Intel RealSense D457 camera to capture RGB and depth frames, with support for simulation mode when no physical camera is available.

## Architecture

The application follows a multi-threaded architecture:

1. **Main Thread**: Initializes the application and manages the overall flow.
2. **Capture Thread**: Captures frames from the camera or generates simulated frames.
3. **Processing Thread**: Processes frames to detect people and prepare hoodie overlays.
4. **Output Thread**: Saves the processed frames to disk.

## Configuration

The application can be configured through command line arguments or by modifying the default values in the code:

```python
# Default configuration values
DEFAULT_DISTANCE_THRESHOLD = 2.0  # meters
DEFAULT_HOODIE_SCALE = 1.0
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_FPS = 30
```

## Extending the Application

The application can be extended in several ways:

1. **Improved Person Tracking**: Implement more sophisticated person tracking to maintain consistent hoodie overlays across frames.
2. **Multiple Hoodie Models**: Support different hoodie models for different people or based on other criteria.
3. **Interactive Controls**: Add real-time controls for adjusting the hoodie model's appearance and position.
4. **Integration with Other Systems**: Connect to e-commerce systems, social media sharing, or other applications.

## Troubleshooting

- **No camera detected**: Use the `--simulate` flag to run in simulation mode.
- **Performance issues**: Reduce the camera resolution or frame rate using the `--width`, `--height`, and `--fps` options.
- **Model loading errors**: Ensure the 3D model file is in a supported format (OBJ, FBX, GLTF) and the path is correct.