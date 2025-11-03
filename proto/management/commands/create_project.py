# import argparse
# import os
# from ..base import BaseCommand, CommandRegistry

# @CommandRegistry.register('create-project')
# class CreateProjectCommand(BaseCommand):
#     help = 'Create a new project with the specified name'

#     def add_arguments(self, parser: argparse.ArgumentParser) -> None:
#         parser.add_argument('name', help='Name of the project')
#         parser.add_argument('--path', default='.', help='Path where to create the project')

#     def handle(self, ) -> None:
#         project_name = options['name']
#         base_path = options['path']
#         project_path = os.path.join(base_path, project_name)

#         try:
#             os.makedirs(project_path, exist_ok=True)
#             # Add your project creation logic here
#             print(f"Created project '{project_name}' at {project_path}")
#         except Exception as e:
#             print(f"Error creating project: {e}")
#             raise


import pickle
import os
from omegaconf import DictConfig, OmegaConf
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from proto.management import BaseCommand, CommandRegistry
from proto.conf import settings


@CommandRegistry.register('create-project')
class CreateProjectCommand(BaseCommand):
    help = 'Create a new project with the specified name'

    def handle(self, cfg: DictConfig) -> None:
        project_name = cfg.project_name
        # project_path = os.path.join(settings.PROJECTS_DIR, project_name)
        # os.makedirs(project_path, exist_ok=True)

        print(project_name)
        