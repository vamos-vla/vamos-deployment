# VAMOS 
## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:vamos-vla/vamos-deployment.git
   cd vamos-deployment
   ```

2. **Set up the environment**:
   ```bash
   conda env create -f environment.yml
   conda activate vamos
   ```

### Quick Test

Test the VLM model with sample images:

```bash
python test_vamos.py
```

Or use the Jupyter notebook for interactive testing:

```bash
jupyter notebook test_vamos.ipynb
```

## ROS Package

VAMOS includes a ROS package. For detailed usage and setup instructions, see [ROS.md](ROS.md).
