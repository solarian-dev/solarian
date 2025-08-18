

# Installation

Create a conda environment:

conda create -n proto python=3.10


Activate the environment:

conda activate proto


Clone the repository:

git clone https://github.com/raulgarreta/teleop.git
cd teleop


Install dependencies:

pip install -e .


# Running

Running server

Within your project directory, run the following command:

python manage.py run-server --conf conf_test


Running robot

Within your project directory, run the following command:

python manage.py run-robot --conf conf_test


Running controller

Within your project directory, run the following command:

python manage.py run-teleop --conf conf_test

