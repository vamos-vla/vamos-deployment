# VAMOS System Setup

VAMOS (Vision-Aided Mobile Object System) is a VLM-based navigation system for robots. Since VLM inference is computationally intensive, there are two deployment modes:

## Deployment Modes

### 1. ROS Node Mode (Robot + Compute)
- **Setup**: Run VLM as ROS node on compute machine connected to robot
- **Launch**: `roslaunch navigate image_navigate_custom.launch` on device with VLM compute
- **Requirements**: ROS network between robot and compute

### 2. HTTP Server Mode (Web Interface)
- **Setup**: Run as HTTP server for web-based control
- **Launch**: `./server/start_server.sh` on server and `roslaunch navigate image_navigate_custom.launch` on robot 
- **Access**: Web interface at `http://your_web_address:8009`
- **Features**: Natural language prompting, model testing

## Web Interface Features

The HTTP server mode provides a web interface with natural language capabilities:

- **Natural Language Steering**: Set global instructions that guide navigation behavior
- **Real-time Prediction**: Upload images and get instant trajectory predictions
- **Model Testing**: Adjust VLM parameters (temperature, tokens, beams) for testing

## Configuration

### HTTP Server Mode Configuration
For HTTP server mode, set these parameters in `navigate_custom.yaml`:
```yaml
use_http_server: true
http_server_url: "http://your_server_ip:8009"
```
#### Model path for HTTP server mode

By default, the HTTP server launches with a model path specified in [`server/start_server.sh`](server/start_server.sh):

You can set a different model path via the `--model_path` argument or by editing the `MODEL_PATH` line in `start_server.sh`. For example:
```bash
./server/start_server.sh --model_path /path/to/your_model
```
Or just edit the value in `server/start_server.sh`:
```bash
MODEL_PATH="/your/local/model/path"
```
