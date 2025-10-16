# VAMOS Package

## Overview

The navigate package includes:
- Navigation node with VLM-based trajectory prediction
- Configurable goal publishers for testing
- Support for different robot configurations (Spot, Hound, Custom)
- Debug capabilities for topic monitoring

## Configuration Files

### Robot-Specific Configurations

- **`navigate_spot.yaml`**: Configuration for Spot robot
- **`navigate_hound.yaml`**: Configuration for Hound robot
- **`navigate_custom.yaml`**: Custom configuration template

### Custom Configuration

The custom configuration (`navigate_custom.yaml`) is designed for testing and development:

```yaml
# Core navigation parameters
image_topic: ""                    # Set your image topic
depth_topic: ""                    # Set your depth topic  
depth_camera_info_topic: ""        # Set your camera info topic
goal_topic: ""                     # Set your goal topic
robot_frame: ""                    # Set your robot frame
global_frame: ""                   # Set your global frame
predictor_type: "vlm"              # Different types only for testing
time_threshold: 60                 # Time in seconds before the VLM times out and replans
debug_subscribers: false           # Set true to debug topic subscribers

# Trajectory predictor parameters
num_waypoints_to_complete: 5       # Number of waypoints to extract
side_offset_for_turn_cam: 0.5     # Side offset for turn camera
behind_offset_for_turn_cam: 0.3   # Behind offset for turn camera
post_rotation_delay: 2.0          # Delay after rotation

# VLM model parameters
model_name_or_path: "/path/to/your/model"  # Path to VLM model
temperature: 0.1                   # Sampling temperature
max_new_tokens: 10                 # Max tokens to generate
num_samples: 50                    # Number of trajectory samples
debug: true                       # Enable debug visualization

# VLM frame parameters
feet_center_frame: "feet_center"  # Robot feet/ground reference frame
```

## Launch Files

### Standard Launch Files
- **`image_navigate_spot.launch`**: Launch navigate node for Spot robot
- **`image_navigate_hound.launch`**: Launch navigate node for Hound robot

### Custom Launch Files
- **`image_navigate_custom.launch`**: Launch navigate node with custom configuration
- **`dummy_goal_publisher_custom.launch`**: Launch dummy goal publisher with custom configuration

## Usage

### Running the Navigation Node

#### For Spot Robot:
```bash
roslaunch navigate image_navigate_spot.launch
```

#### For Hound Robot:
```bash
roslaunch navigate image_navigate_hound.launch
```

#### For Custom Configuration:
```bash
roslaunch navigate image_navigate_custom.launch
```

### Running the Dummy Goal Publisher

The dummy goal publisher is a testing utility that continuously publishes navigation goals to test the navigation system without requiring manual goal input. It publishes `geometry_msgs/PoseStamped` messages at regular intervals with predefined goal positions.

#### What is the Dummy Goal Publisher?

The dummy goal publisher is a ROS node that:
- Publishes navigation goals at 10Hz to a configurable topic
- Uses predefined goal positions (x=5.0, y=0.0, z=0.0) with identity orientation
- Automatically updates timestamps for each published goal
- Can be configured for different robot types and coordinate frames
- Essential for testing navigation systems without manual intervention

#### Launch File Usage (Recommended):

The easiest way to run the dummy goal publisher is using the provided launch file:

```bash
# Using custom configuration
roslaunch navigate dummy_goal_publisher_custom.launch
```

This launch file automatically loads the configuration from `navigate_custom.yaml` and sets up the appropriate parameters.

### Complete Testing Setup

To test the navigation system with custom configuration:

1. **Configure your topics** in `navigate_custom.yaml`:
   ```yaml
   image_topic: "/your_camera/image_raw"
   depth_topic: "/your_camera/depth/image_raw"
   depth_camera_info_topic: "/your_camera/depth/camera_info"
   goal_topic: "/custom/goal"
   robot_frame: "base_link"
   ```

2. **Launch the navigation node**:
   ```bash
   roslaunch navigate image_navigate_custom.launch
   ```

3. **Launch the dummy goal publisher** (using the new launch file):
   ```bash
   roslaunch navigate dummy_goal_publisher_custom.launch
   ```

   Or using direct node execution:
   ```bash
   rosrun navigate dummy_goal_publisher.py _goal_topic:=/custom/goal
   ```

## Parameters

### Core Navigation Parameters

- **`image_topic`**: ROS topic for RGB images (e.g., `/camera/color/image_raw`)
- **`depth_topic`**: ROS topic for depth images (e.g., `/camera/aligned_depth_to_color/image_raw`)
- **`depth_camera_info_topic`**: ROS topic for camera calibration info (e.g., `/camera/aligned_depth_to_color/camera_info`)
- **`goal_topic`**: ROS topic for navigation goals (e.g., `/spot/goal`, `/hound/goal`)
- **`robot_frame`**: TF frame of the robot base (e.g., `base_link`)
- **`global_frame`**: Global reference frame (e.g., `map`, `vision`)
- **`predictor_type`**: Type of trajectory predictor ("vlm", "dummy", "base")
- **`time_threshold`**: Timeout for VLM processing (seconds)
- **`debug_subscribers`**: Enable debug logging for topic subscribers

### Trajectory Predictor Parameters

- **`num_waypoints_to_complete`**: Number of waypoints to extract from predicted trajectories (default: 5)
- **`side_offset_for_turn_cam`**: Side offset for turn camera positioning (meters, default: 0.5)
- **`behind_offset_for_turn_cam`**: Behind offset for turn camera positioning (meters, default: 0.3)
- **`post_rotation_delay`**: Delay after rotation before continuing (seconds, default: 2.0)
- **`path_topic`**: Topic for global planner path (`/global_planner/planned_path`)
- **`waypoint_reached_topic`**: Topic for waypoint completion (`/carrot_picker/waypoint_reached_index`)
- **`mission_complete_topic`**: Topic for mission completion (`/nav_manager/mission_complete`)

### VLM Model Parameters

- **`model_name_or_path`**: Path to the VLM model directory
- **`temperature`**: Sampling temperature for text generation (default: 0.1)
- **`max_new_tokens`**: Maximum number of new tokens to generate (default: 10)
- **`num_samples`**: Number of trajectory samples to generate (default: 50)
- **`top_k`**: Top-k sampling parameter (default: 0)
- **`num_beams`**: Number of beams for beam search (default: 1)
- **`debug`**: Enable debug visualization (default: true)

### VLM Topic Parameters

- **`debug_image_topic`**: Topic for debug trajectory visualization (`/vlm_ros/image_predicted_trajectories`)
- **`value_map_image_topic`**: Topic for value map visualization (`/vlm_ros/value_map_trajectories`)
- **`predicted_paths_2d_topic`**: Topic for 2D predicted paths (`/vlm_ros/predicted_paths_2d`)
- **`predicted_paths_3d_topic`**: Topic for 3D predicted paths (`/vlm_ros/predicted_paths_3d`)
- **`predicted_values_topic`**: Topic for predicted trajectory values (`/vlm_ros/predicted_values`)

### VLM Frame Parameters

- **`feet_center_frame`**: TF frame for robot feet/ground reference (default: `feet_center`)

### Dummy Goal Publisher Parameters

- **`goal_topic`**: Topic to publish goals to (default: `/spot/goal`)
- **`global_frame`**: Frame ID for published goals (default: `map`)

## VLM Model Setup

### Downloading VLM Models

The system uses Hugging Face models for VLM-based trajectory prediction. Use the provided download script to set up your VLM model:

#### Basic Usage:
```bash
# Download a model (example with Paligemma2 model)
python3 vlm_ros/src/vlm_ros/dowload_hf_model.py --model_id "mateoguaman/paligemma2-3b-pt-224-sft-lora-magicsoup"
```

#### Advanced Usage:
```bash
# Download to custom directory
python3 vlm_ros/src/vlm_ros/dowload_hf_model.py --model_id "your-model-id" --output-dir "/path/to/models"

# Skip modification prompts (for automated setups)
python3 vlm_ros/src/vlm_ros/dowload_hf_model.py --model_id "your-model-id" --no-modify
```

#### What the Download Script Does:

The `dowload_hf_model.py` script:
- Downloads Hugging Face models using `huggingface_hub`
- Creates organized model directories under `./models/` (or custom path)
- Automatically detects and handles `adapter_config.json` files
- Provides interactive modification of adapter configurations
- Removes problematic keys (like `eva_config`) that may cause compatibility issues
- Allows manual editing of configuration files
- Uses actual files instead of symlinks for better compatibility

#### Model Configuration:

After downloading, update your configuration file with the model path:

```yaml
# In your navigate_custom.yaml or other config file
model_name_or_path: "/path/to/your/downloaded/model"
```