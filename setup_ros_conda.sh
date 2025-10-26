#!/bin/bash

# Setup ROS Noetic within conda environment using robostack
# This script should be run after activating the conda environment

set -e

echo "Setting up ROS Noetic in conda environment..."

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "vamos" ]]; then
    echo "Error: Please activate the 'vamos' conda environment first:"
    echo "conda activate vamos"
    exit 1
fi

# Source ROS setup for robostack
echo "Sourcing ROS setup..."
if [ -f "$CONDA_PREFIX/etc/ros/ros_setup.sh" ]; then
    source $CONDA_PREFIX/etc/ros/ros_setup.sh
elif [ -f "$CONDA_PREFIX/share/ros/setup.sh" ]; then
    source $CONDA_PREFIX/share/ros/setup.sh
else
    echo "ROS setup file not found. Trying to source from robostack..."
    # Try to source from robostack location
    if [ -f "$CONDA_PREFIX/etc/ros/noetic/setup.bash" ]; then
        source $CONDA_PREFIX/etc/ros/noetic/setup.bash
    else
        echo "Warning: ROS setup not found in conda environment."
        echo "Make sure robostack-noetic packages are installed correctly."
        exit 1
    fi
fi

# Initialize rosdep if not already done
if [ ! -f "$HOME/.ros/rosdep/sources.cache" ]; then
    echo "Initializing rosdep..."
    rosdep init --rosdistro noetic || true
    rosdep update
fi

# Create a workspace if it doesn't exist
WORKSPACE_DIR="$HOME/vamos_ws"
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "Creating catkin workspace at $WORKSPACE_DIR"
    mkdir -p "$WORKSPACE_DIR/src"
    cd "$WORKSPACE_DIR"
    
    # Use colcon instead of catkin_make for better compatibility
    echo "Building workspace with colcon..."
    colcon build --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5
else
    echo "Workspace already exists at $WORKSPACE_DIR"
    cd "$WORKSPACE_DIR"
    echo "Rebuilding workspace..."
    colcon build --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5
fi

echo "ROS Noetic setup completed in conda environment!"
echo ""
echo "To use ROS in this environment:"
echo "1. Activate the conda environment: conda activate vamos"
echo "2. Source ROS: source $CONDA_PREFIX/etc/ros/noetic/setup.bash"
echo "3. Source your workspace: source $WORKSPACE_DIR/install/setup.bash"
echo ""
echo "You can add these to your ~/.bashrc for automatic setup:"
echo "conda activate vamos"
echo "source $CONDA_PREFIX/etc/ros/noetic/setup.bash"
echo "source $WORKSPACE_DIR/install/setup.bash"
