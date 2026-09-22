import time
import threading
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils import get_class
from solarian.core.configs import ReplayConfig
from solarian.utils import log_say
from solarian.core.dataset import Dataset
from solarian.core import Observation
from solarian.core.data import PosData, SpeedData, ImageData, BoundingBoxData, TemperatureData
from solarian.core import Action



@CommandRegistry.register('replay', 'replay')
class ReplayCommand(BaseCommand):
    help = "Replay a recorded dataset's actions back onto the robot"
    description = """
    Prints what a recorded LeRobot dataset contains, then feeds its recorded actions
    back to the robot so an episode runs again without an operator. Use it to check
    that a dataset drives the hardware the way it was captured.

    The robot and the relay have to be up, and the scene has to start where the
    episode started: the actions are played as recorded, with no feedback.
    conf/replay.yaml holds repo_id, episodes, reset_time_s and reset_period.
    """
    examples = (
        "solarian replay",
        "solarian replay repo_id=my-user/cup-arrange episodes=[3]",
    )

    def get_features_lerobot_format(self):

        from solarian.conf import settings
        
        features = {
            "observation.state": {
                "shape": (0,),
                "names": [],
                "dtype": "float32",
            },
            "observation.environment_state": {
                "shape": (0,),
                "names": [],
                "dtype": "float32",
            },
            "action": {
                "shape": (0,),
                "names": [],
                "dtype": "float32",
            }
        }
        num_cameras = 0

        print("Observation:")
        

        for module_name, sensor_name, data_name, data_spec in Observation.items():
            if type(data_spec) in [PosData, SpeedData, BoundingBoxData, TemperatureData]:
                lerobot_group = settings.LEROBOT_GROUP_NAMES[module_name][sensor_name][data_name]
                features[lerobot_group]["shape"] = tuple(a + b for a, b in zip(features[lerobot_group]["shape"], data_spec.shape))
                features[lerobot_group]["names"].extend(".".join([module_name, sensor_name, data_name]))
            elif type(data_spec) == ImageData:
                features[f"observation.images.{'.'.join([module_name, sensor_name, data_name])}"] = {
                    "shape": self.cfg.video_res,
                    "names": ["height", "width", "channels"],
                    "info": None,
                    "dtype": "video"
                }
                num_cameras += 1
            else:
                raise ValueError(f"Unknown data type: {data_name}")

        for module_name, actuator_name, data_name, data_spec in Action.items():
            if type(data_spec) in [PosData, SpeedData]:
                lerobot_group = settings.LEROBOT_GROUP_NAMES[module_name][actuator_name][data_name]
                features[lerobot_group]["shape"] = tuple(a + b for a, b in zip(features[lerobot_group]["shape"], data_spec.shape))
                features[lerobot_group]["names"].extend(".".join([module_name, actuator_name, data_name]))
            else:
                raise ValueError(f"Unknown data type: {data_name}")
        return features, num_cameras


    def handle(self, cfg: DictConfig) -> None:
        # Imported here so `solarian replay --help` does not need lerobot installed.
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        # Instantiate config
        cfg = ReplayConfig(**OmegaConf.to_object(cfg))
        self.cfg = cfg

        ds_meta = LeRobotDatasetMetadata(cfg.repo_id)

        print(f"Total number of episodes: {ds_meta.total_episodes}")
        print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
        print(f"Frames per second used during data collection: {ds_meta.fps}")
        print(f"Robot type: {ds_meta.robot_type}")
        print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")
        print("Chunk size: ", ds_meta.chunks_size)
        print("Features: ", ds_meta.features)


        
        

        # Create controller instance
        ControllerClass = get_class("lekiwi.replay_controller.ReplayController")
        controller = ControllerClass(cfg)
        controller.connect()
        controller.initialize()
        # controller.start()

        # features, num_cameras = self.get_features_lerobot_format()


        # reset the environment
        if cfg.reset_time_s > 0:
            # log_say("Reset the environment", cfg.play_sounds, blocking=True)
            # controller.control_loop(episode_time_s=cfg.reset_time_s)
            # print("Environment reset")

            log_say(f"Start replaying in three", cfg.play_sounds, blocking=True)
            time.sleep(0.1)
            log_say(f"two", cfg.play_sounds, blocking=True)
            time.sleep(0.1)
            log_say(f"one", cfg.play_sounds, blocking=True)
            time.sleep(0.1)

        

        save_threads = {}
        for i, episode_number in enumerate(cfg.episodes):

            controller.load_dataset_episode(episode_number)
            
            log_say(f"Replaying episode {episode_number}", cfg.play_sounds)
            controller.start()

            # if not controller.events["stop_recording"] and (
            #     (i < cfg.num_episodes and j == len(cfg.tasks) - 1 and (i % cfg.reset_period == 0)) or controller.events["rerecord_episode"]
            # ):
            #     reset_time_s = cfg.reset_time_s
            #     if (i % cfg.reset_period == 0):
            #         reset_time_s = cfg.reset_time_s
                    
            #     # reset the environment
            #     if reset_time_s > 0:
            #         log_say("Reset the environment", cfg.play_sounds, blocking=True)
            #         controller.control_loop(episode_time_s=reset_time_s)
            #     # reset_environment(client, events, reset_time_s, args.fps)

            # if controller.events["rerecord_episode"]:
            #     log_say("Re-record episode", cfg.play_sounds)
            #     controller.events["rerecord_episode"] = False
            #     controller.events["exit_early"] = False
            #     # dataset.clear_episode_buffer()
            #     continue

            # print("-------------->Saving episode...")
            # if str(j) in save_threads:
            #     save_threads[str(j)].join()
            # save_thread = threading.Thread(target=dataset.save_episode, daemon=True)
            # save_thread.start()
            # save_threads[str(j)] = save_thread
            # save_threads.append(save_thread)
            # dataset.save_episode()
            # print("-------------->Episode saved.")
            
            # if controller.events["stop_recording"]:
            #     break
        
        # log_say("Saving episodes", cfg.play_sounds, blocking=False)

        print("Disconnecting controller")
        controller.disconnect()

 