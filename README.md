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

python manage.py run-server --conf conf_test
```

### Running robot

Within your project directory, run the following command:

```bash
python manage.py run-robot --conf conf_test
```

### Running controller

Within your project directory, run the following command:

```bash
python manage.py run-teleop --conf conf_test
```


