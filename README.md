# Solarian

A robotics control and teleoperation framework.

## Installation

### Create a conda environment:

```bash
conda create -n solarian python=3.10
```

### Activate the environment:

```bash
conda activate solarian
```

### Install FFmpeg

```bash
conda install ffmpeg=7.1.1 -c conda-forge
```


### Install Lerobot

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```


### Clone Solarian repository

```bash
git clone https://github.com/raulgarreta/solarian.git
cd solarian
```

### Install dependencies:

With pip:

```bash
pip install -e .
```

With uv (faster; use the same conda env):

```bash
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh
conda activate solarian   # or your env name
uv pip install -e .
```

uv uses whichever Python is active, so packages are installed into your conda env.



### macOS (Darwin) on Apple Silicon (arm64)

You will need to install separate extra requirements to install atomics library from source:

```bash
pip install -r requirements-macos-arm64.txt
```

### Install xformers

```bash
pip install xformers==0.0.31.post1
```

## Running

### Running server

Within your project directory, run the following command:

```bash
python manage.py run-server
```

You can configure to run the server in a different machine by setting the `ip` parameter in the `conf/connection/localhost.yaml` file.

### Running robot

Within your project directory, run the following command:

```bash
python manage.py run-robot
```

### Running controller

Within your project directory, run the following command:

```bash
python manage.py run-controller
```

### Recording a dataset

Within your project directory, run the following command:

```bash
python manage.py record --conf conf/record_eval
```

### Converting a dataset to Lerobot format

Within your project directory, run the following command:

```bash
python manage.py dataset2lerobot

## Other commands

sudo chmod 666 /dev/ttyACM1


python lerobot/find_port.py


# First, install v4l-utils if not already installed
sudo apt install v4l-utils

# List available cameras and their capabilities
v4l2-ctl --list-devices

# Check supported formats and frame rates for your camera (replace /dev/video0 with your device)
v4l2-ctl -d /dev/video0 --list-formats-ext
```

### Training a model

First you should create a config file in the `conf/train` directory. Then you can run the following command:

```bash
python manage.py train --conf conf/your_train_config.yaml
```

## Useful commands

### Watch GPU usage

```bash
watch -n 2 nvidia-smi
```

### find motor port

```bash
python lerobot/find_port.py
```
