# VAMOS Package

## Installation

### Download VLM Models

The system uses Hugging Face models for VLM-based trajectory prediction. Download the VAMOS model using the provided script:

```bash
# Download the example Paligemma2 model
python3 vlm_ros/src/vlm_ros/dowload_hf_model.py --model_id "mateoguaman/paligemma2-3b-pt-224-sft-lora-magicsoup"
```

#### What the Download Script Does:

The `dowload_hf_model.py` script:
- Downloads Hugging Face models using `huggingface_hub`
- Creates organized model directories under `./models/`
- Automatically detects and handles `adapter_config.json` files
- Provides interactive modification of adapter configurations
- Removes problematic keys (like `eva_config`) that may cause compatibility issues
- Allows manual editing of configuration files
- Uses actual files instead of symlinks for better compatibility

### Download Value Function Models

Download the value function model for terrain evaluation:

```bash
# Create the model directory
mkdir -p value_functions

# Download the rough terrain critic model
# (Replace with your actual download command or instructions)
wget -O value_functions/rough_critic.onnx "your-model-url"
```

## Configuration

### Custom Configuration File

The system uses `navigate_custom.yaml` for custom robot configurations. This file allows you to configure all aspects of the navigation system for your specific robot setup.

#### Core Navigation Parameters

Edit these parameters in `navigate_custom.yaml`:

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
```

#### VLM Model Configuration

Configure your VLM model path:

```yaml
# VLM parameters
model_name_or_path: "/path/to/your/downloaded/model"  # Path to VLM model
temperature: 0.1                   # Sampling temperature
max_new_tokens: 10                 # Max tokens to generate
num_samples: 50                    # Number of trajectory samples
debug: true                       # Enable debug visualization
```

#### Value Function Configuration

Configure the value function model:

```yaml
# Value publisher parameters
imu_topic: ""                      # IMU data topic
heightmap_topic: ""                # Height map data topic
nn_path: ""                        # Path to value function model
value_map_topic: "/vlm_ros/value_map"
value_map_image_topic: "/vlm_ros/value_map_image"
```

#### Topic Configuration

Configure all ROS topics for your robot:

```yaml
# VLM topic parameters
debug_image_topic: "/vlm_ros/image_predicted_trajectories"
value_map_image_topic_trajectories: "/vlm_ros/value_map_trajectories"
predicted_paths_2d_topic: "/vlm_ros/predicted_paths_2d"
predicted_paths_3d_topic: "/vlm_ros/predicted_paths_3d"
predicted_values_topic: "/vlm_ros/predicted_values"
```

## Launch Files

### Custom Launch File

Use the custom launch file to run the navigation system with your configuration:

```bash
# Launch the navigation system with custom configuration
roslaunch navigate image_navigate_custom.launch
```

This launch file:
- Loads parameters from `navigate_custom.yaml`
- Starts the VLM inference node
- Starts the navigation node with your custom configuration
- Includes the value publisher for terrain evaluation

### Complete Testing Setup

To test the navigation system with custom configuration:

1. **Configure your topics** in `navigate_custom.yaml`:
   ```yaml
   image_topic: "/your_camera/image_raw"
   depth_topic: "/your_camera/depth/image_raw"
   depth_camera_info_topic: "/your_camera/depth/camera_info"
   goal_topic: "/custom/goal"
   robot_frame: "base_link"
   global_frame: "map"
   ```

2. **Launch the navigation node**:
   ```bash
   roslaunch navigate image_navigate_custom.launch
   ```

3. **Launch the dummy goal publisher** (for testing):
   ```bash
   roslaunch navigate dummy_goal_publisher_custom.launch
   ```

## Dummy Goal Publisher

The dummy goal publisher is a testing utility that continuously publishes navigation goals to test the navigation system without requiring manual goal input.

### What is the Dummy Goal Publisher?

The dummy goal publisher is a ROS node that:
- Publishes navigation goals at 10Hz to a configurable topic
- Uses predefined goal positions (x=5.0, y=0.0, z=0.0) with identity orientation
- Automatically updates timestamps for each published goal
- Can be configured for different robot types and coordinate frames
- Essential for testing navigation systems without manual intervention

### Usage

The easiest way to run the dummy goal publisher is using the provided launch file:

```bash
# Using custom configuration
roslaunch navigate dummy_goal_publisher_custom.launch
```

This launch file automatically loads the configuration from `navigate_custom.yaml` and sets up the appropriate parameters.