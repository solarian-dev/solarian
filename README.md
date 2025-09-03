# Proto

A robotics control and teleoperation framework.

## Installation

### Create a conda environment:

```bash
conda create -n proto python=3.10
```

### Activate the environment:

```bash
conda activate proto
```

### Install FFmpeg

```bash
conda install ffmpeg=7.1.1 -c conda-forge
```

### Clone the repository:

```bash
git clone https://github.com/raulgarreta/teleop.git
cd teleop
```

### Install dependencies:

```bash
pip install -e .
```

## Running

### Running server

Within your project directory, run the following command:

```bashconda create -n proto python=3.10

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