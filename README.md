# Hoodie Overlay Application

A real-time application that detects people using the Intel RealSense D457 camera and overlays a 3D hoodie model on them.

## Features

- **Person Detection**: Uses OpenCV DNN with MobileNet SSD to detect people in real-time
- **Distance Calculation**: Calculates the distance of detected people using depth data
- **3D Hoodie Overlay**: Overlays a 3D hoodie model on detected people within a specified distance threshold
- **Adaptive Performance**: Automatically adjusts processing parameters to maintain target frame rate
- **Real-time Statistics**: Displays FPS and processing time information on-screen
- **Configuration System**: Supports command-line arguments and configuration files

## Requirements

- Python 3.6+
- Intel RealSense D457 camera (or compatible)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/hoodie-overlay-app.git
   cd hoodie-overlay-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the Intel RealSense SDK:
   - Follow the instructions at [https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)
   - For Windows users, you can download the SDK from [https://www.intelrealsense.com/sdk-2/](https://www.intelrealsense.com/sdk-2/)

## Usage

### Basic Usage

```bash
python optimized_hoodie_overlay_app.py
```

### Run in Simulation Mode (no camera required)

```bash
python optimized_hoodie_overlay_app.py --simulate
```

### Adjust Distance Threshold and Hoodie Scale

```bash
python optimized_hoodie_overlay_app.py --threshold 1.5 --scale 1.2
```

### Specify Custom 3D Model

```bash
python optimized_hoodie_overlay_app.py --model path/to/hoodie_model.obj
```

### Adjust Performance Parameters

```bash
python optimized_hoodie_overlay_app.py --target-fps 30 --downscale 1.5 --detection-interval 2
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--simulate` | Run in simulation mode (no physical camera required) | False |
| `--threshold THRESHOLD` | Distance threshold in meters | 2.0 |
| `--scale SCALE` | Hoodie model scale factor | 1.0 |
| `--model MODEL` | Path to the 3D model file | None |
| `--width WIDTH` | Camera width in pixels | 1280 |
| `--height HEIGHT` | Camera height in pixels | 720 |
| `--fps FPS` | Camera frames per second | 30 |
| `--output-dir OUTPUT_DIR` | Directory to save output files | output |
| `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}` | Logging level | INFO |
| `--config-file CONFIG_FILE` | Configuration file path | hoodie_overlay_config.json |
| `--target-fps TARGET_FPS` | Target frames per second | 25.0 |
| `--frame-skip FRAME_SKIP` | Number of frames to skip | 0 |
| `--downscale DOWNSCALE` | Factor to downscale images for processing | 1.0 |
| `--detection-interval DETECTION_INTERVAL` | Process every Nth frame for detection | 1 |

## Configuration File

You can store your preferred settings in a JSON configuration file:

```json
{
    "distance_threshold": 1.8,
    "hoodie_scale": 1.2,
    "target_fps": 30.0,
    "downscale_factor": 1.5,
    "detection_interval": 2
}
```

The application will automatically save your current settings to the configuration file when it exits.

## Performance Optimization

The application includes several performance optimization features:

1. **Adaptive Frame Skipping**: Automatically skips frames when needed to maintain target FPS
2. **Resolution Adjustment**: Dynamically adjusts processing resolution based on performance
3. **Detection Interval**: Performs person detection only on selected frames to reduce CPU usage
4. **Performance Monitoring**: Continuously monitors FPS and processing times to adjust parameters

## Output

The application generates the following output files in the specified output directory:

- `frame_XXXX_rgb.jpg`: RGB image with person detection and hoodie overlay
- `frame_XXXX_depth.jpg`: Depth image with colormap applied
- `frame_XXXX_combined.jpg`: Combined RGB and depth images

## Architecture

The application follows a multi-threaded architecture:

1. **Main Thread**: Initializes the application and manages the overall flow
2. **Capture Thread**: Captures frames from the camera or generates simulated frames
3. **Processing Thread**: Processes frames to detect people and prepare hoodie overlays
4. **Output Thread**: Saves the processed frames to disk

## Troubleshooting

### No Camera Detected

If no RealSense camera is detected, try:
- Ensuring the camera is properly connected
- Running with the `--simulate` flag to test in simulation mode
- Reinstalling the RealSense SDK

### Performance Issues

If the application runs slowly:
- Reduce the camera resolution with `--width` and `--height`
- Increase the downscale factor with `--downscale`
- Increase the detection interval with `--detection-interval`
- Increase the frame skip with `--frame-skip`

### Model Loading Errors

If there are issues loading the 3D model:
- Ensure the model file is in a supported format (OBJ, FBX, GLTF)
- Check that the path to the model file is correct
- Try running without a custom model to use the built-in placeholder

## Logging

The application logs information to both the console and a file (`hoodie_overlay.log`). You can adjust the logging level with the `--log-level` option.

## License

This project is licensed under the MIT License - see the LICENSE file for details.