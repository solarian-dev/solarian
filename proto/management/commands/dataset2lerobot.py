import pickle
import os
from omegaconf import DictConfig, OmegaConf
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from proto.management import BaseCommand, CommandRegistry
from proto.conf import settings


@CommandRegistry.register('dataset2lerobot', 'dataset2lerobot')
class Dataset2LeRobotCommand(BaseCommand):
    help = 'Convert dataset to lerobot dataset.'

    def handle(self, cfg: DictConfig) -> None:

        features = pickle.load(open(os.path.join(cfg.input_root, "pickles", "features.pkl"), "rb"))
        
        # create lerobot dataset
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.output_root,
            robot_type=settings.ROBOT_NAME,
            features=features,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_cameras*cfg.num_image_writer_threads_per_camera,
        )

        # iterate over each episode pickle file and add it as a new episode to the lerobot dataset
        for episode_file in os.listdir(os.path.join(cfg.input_root, "pickles")):
            if episode_file.endswith(".pkl") and not episode_file.startswith("features.pkl"):
                episode = pickle.load(open(os.path.join(cfg.input_root, "pickles", episode_file), "rb"))
                for frame in episode:
                    dataset.add_frame(frame["frame"], frame["task"])
                dataset.save_episode()

        if cfg.push_to_hub:
            dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

