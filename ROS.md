# VAMOS Package

## Installation

### Install Packages and Download VLM

First follow the install steps in [README.md](README.md).
IMPORTANT: Make sure to find and set the vlm path from the model download in the config file.

```bash
chmod +x ./setup.sh
./setup.sh
cd vamos_ws
catkin_make
sourc devel/setup,bash
```
### Download Value Function Models

The Value Function Models are hosted on [Hugging Face](https://huggingface.co/collections/mateoguaman/vamos-a-hierarchical-vision-language-action-model-for-capab). 
Download the desired model files, then update the `nn_path` parameter in your configuration file (`navigate_custom.yaml`) to point to the downloaded model location.

## Configuration

### Custom Configuration File

The system uses `navigate_custom.yaml` for custom robot configurations.

#### Core Navigation Parameters

Edit these parameters in `navigate_custom.yaml`:

```yaml
# Core navigation parameters
image_topic: ""
depth_topic: ""
depth_camera_info_topic: ""
goal_topic: ""                     # Goal should be in global frame
robot_frame: ""
global_frame: ""
predictor_type: "vlm"              # Different types only for testing, always use vlm
time_threshold: 60                 # Time in seconds before the VLM times out and replans
debug_subscribers: false           # Set true to debug topic subscribers
```

#### VLM Model Configuration

Configure your VLM model path:

```yaml
# VLM parameters
model_name_or_path: "/path/to/your/downloaded/model"  # Path to VLM model
temperature: 0.1
max_new_tokens: 10
num_samples: 50
debug: true                       # Enable debug visualization
```

#### Value Function Configuration

Configure the value function model:

```yaml
# Value publisher parameters
imu_topic: ""
heightmap_topic: ""
nn_path: ""                        # Path to value function model
value_map_topic: "/vamos/value_map"
value_map_image_topic: "/vamos/value_map_image"
```

#### Topic Configuration

Configure all ROS topics for your robot:

```yaml
# VLM topic parameters
debug_image_topic: "/vamos/image_predicted_trajectories"
value_map_image_topic_trajectories: "/vamos/value_map_trajectories"
predicted_paths_2d_topic: "/vamos/predicted_paths_2d"
predicted_paths_3d_topic: "/vamos/predicted_paths_3d"
predicted_values_topic: "/vamos/predicted_values"
```

### Complete Testing Setup

To test the navigation system with custom configuration:

1. **Configure your topics** in `navigate_custom.yaml`

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

## State Machine Documentation

The VAMOS navigation system uses a state machine to manage robot navigation behavior.

### Navigation States

The system operates through six distinct states:

#### 1. IDLE
- **Purpose**: Initial state and reset state
- **Behavior**: 
  - Resets waypoint tracking counters
  - Clears previous trajectory data
- **Transitions**:
  - `MISSION_COMPLETE` → If mission is finished
  - `ROTATING_TO_VIEW_GOAL` → If goal is not visible in camera
  - `AWAITING_VLM_PREDICTION` → If goal is visible in camera

#### 2. ROTATING_TO_VIEW_GOAL
- **Purpose**: Robot turns to bring the goal into camera view
- **Behavior**:
  - Generates a turn trajectory when the goal is out of view.
    - The trajectory consists of a single waypoint that is placed slightly behind the camera
      and offset laterally to one side, opposite the sign of the goal's x-position in the camera frame.
- **Transitions**:
  - `WAITING_FOR_CAMERA` → When goal becomes visible
  - Continues in same state → If goal still not visible

#### 3. WAITING_FOR_CAMERA
- **Purpose**: Allows camera to stabilize after rotation
- **Behavior**:
  - Waits for configurable delay (`post_rotation_delay` parameter)
- **Transitions**:
  - `AWAITING_VLM_PREDICTION` → After stabilization delay

#### 4. AWAITING_VLM_PREDICTION
- **Purpose**: Generates navigation trajectory using VLM
- **Behavior**:
  - Calls VLM to predict 3D waypoints
- **Transitions**:
  - `EXECUTING_WAYPOINTS` → If VLM prediction successful
  - `IDLE` → If VLM prediction fails

#### 5. EXECUTING_WAYPOINTS
- **Purpose**: Robot follows the predicted trajectory
- **Behavior**:
  - Monitors waypoint completion progress
  - Tracks time since last VLM prediction
  - Publishes empty path when segment complete or timeout
- **Transitions**:
  - `IDLE` → When waypoint segment completed or timeout reached

#### 6. MISSION_COMPLETE
- **Purpose**: Final state when mission is finished
- **Behavior**:
  - Stops all trajectory prediction
  - Maintains mission complete status
- **Transitions**:
  - Stays in `MISSION_COMPLETE` → No further transitions