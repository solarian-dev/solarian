import time
import threading
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils import get_class
from solarian.core.configs import RecordConfig
from solarian.utils import log_say
from solarian.core.dataset import Dataset
from solarian.core.data import PosData, SpeedData, ImageData, BoundingBoxData, TemperatureData
from solarian.core import Observation
from solarian.core import Action



@CommandRegistry.register('record', 'record')
class RecordCommand(BaseCommand):
    help = "Teleoperate and record episodes into a LeRobot dataset"
    description = """
    Teleoperates the robot like run-controller and writes what happens to a LeRobot
    dataset: every observation, every action and one video stream per camera, cut
    into episodes.

    The robot (`solarian run-robot`) and the relay (`solarian run-server`) have to
    be up. conf/record.yaml holds the run: repo_id, num_episodes, episode_time_s,
    reset_time_s, fps, video_res, tasks, and push_to_hub / tags / private for the
    Hub upload. start_episode resumes into an existing dataset rather than
    overwriting it.
    """
    examples = (
        "solarian record",
        "solarian record repo_id=my-user/cup-arrange num_episodes=50",
        "solarian record num_episodes=10 start_episode=51 episode_time_s=20",
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

        for obs_name, obs in Observation.get_sensor_obs().items():
            if obs.data_class in [PosData, SpeedData, BoundingBoxData, TemperatureData]:
                lerobot_group = settings.LEROBOT_GROUP_NAMES[obs_name]
                features[lerobot_group]["shape"] = tuple(a + b for a, b in zip(features[lerobot_group]["shape"], obs.shape))
                features[lerobot_group]["names"].append(obs_name)
            elif obs.data_class == ImageData:
                features[f"observation.images.{obs_name}"] = {
                    "shape": self.cfg.video_res,
                    "names": ["height", "width", "channels"],
                    "info": None,
                    "dtype": "video"
                }
                num_cameras += 1
            else:
                raise ValueError(f"Unknown data type: {obs_name}")

        for module_name, actuator_name, data_name, data_spec in Action.items():
            if type(data_spec) in [PosData, SpeedData]:
                lerobot_group = settings.LEROBOT_GROUP_NAMES[module_name][actuator_name][data_name]
                features[lerobot_group]["shape"] = tuple(a + b for a, b in zip(features[lerobot_group]["shape"], data_spec.shape))
                features[lerobot_group]["names"].extend(".".join([module_name, actuator_name, data_name]))
            else:
                raise ValueError(f"Unknown data type: {data_name}")
        return features, num_cameras


    def handle(self, cfg: DictConfig) -> None:
        from solarian.conf import settings

        # print(cfg)
        
        # Instantiate config
        cfg = RecordConfig(**OmegaConf.to_object(cfg))
        self.cfg = cfg

        # Create controller instance
        ControllerClass = get_class(settings.CONTROLLER_CLASS)
        controller = ControllerClass(cfg)
        controller.connect()
        controller.initialize()

        features, num_cameras = self.get_features_lerobot_format()

        # Create empty dataset or load existing saved episodes
        # sanity_check_dataset_name(args.repo_id, None)
        datasets = {}
        for task in cfg.tasks:
            dataset = Dataset(
                cfg.repo_id + "_" + task.replace(" ", "_"),
                cfg.fps,
                root='./datasets/' + cfg.repo_id + "_" + task.replace(" ", "_"),
                robot_type=settings.ROBOT_NAME,
                features=features,
                use_videos=cfg.video,
                image_writer_processes=cfg.num_image_writer_processes,
                image_writer_threads=num_cameras * cfg.num_image_writer_threads_per_camera,
                start_episode=cfg.start_episode,
            )
            # datasets[task] = dataset
            # dataset = LeRobotDataset.create(
            #     cfg.repo_id + "_" + task.replace(" ", "_"),
            #     cfg.fps,
            #     root='./datasets/' + cfg.repo_id + "_" + task.replace(" ", "_"),
            #     # robot=None,
            #     robot_type=settings.ROBOT_NAME,
            #     features=features,
            #     use_videos=cfg.video,
            #     image_writer_processes=cfg.num_image_writer_processes,
            #     image_writer_threads=num_cameras * cfg.num_image_writer_threads_per_camera,
            # )
            datasets[task] = dataset

        print("Datasets created")

        # # Create events dictionary directly instead of using init_keyboard_listener
        # events = {
        #     "exit_early": False,
        #     "rerecord_episode": False, 
        #     "stop_recording": False,
        #     "track all": False,
        #     "verbose": False,
        #     "prompt": None,
        #     "is_prompting": False,
        #     "reset_policy": False
        # }
        
        # Extend the controller's keyboard listener with additional handlers for recording
        # Store original on_press to chain to it
        # original_on_press = controller.on_press
        
        # Create a new on_press function that handles recording events and calls the original
        # def extended_on_press(key):
        #     try:
        #         # Handle recording-specific keys
        #         if key == keyboard.Key.right:
        #             print("Right arrow key pressed. Exiting loop...")
        #             events["exit_early"] = True
        #         elif key == keyboard.Key.left:
        #             print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
        #             events["rerecord_episode"] = True
        #             events["exit_early"] = True
        #         elif key == keyboard.Key.esc:
        #             print("Escape key pressed. Stopping data recording...")
        #             events["stop_recording"] = True
        #             events["exit_early"] = True
                
        #         # Call the original on_press handler
        #         return original_on_press(key)
        #     except Exception as e:
        #         print(f"Error handling key press: {e}")
        #         return True
        
        # # Replace the on_press handler with our extended version
        # controller.on_press = extended_on_press

        # reset the environment
        if cfg.reset_time_s > 0:
            log_say("Reset the environment", cfg.play_sounds, blocking=True)
            controller.control_loop(episode_time_s=cfg.reset_time_s)
            print("Environment reset")

            log_say(f"Start recording in three", cfg.play_sounds, blocking=True)
            time.sleep(0.1)
            log_say(f"two", cfg.play_sounds, blocking=True)
            time.sleep(0.1)
            log_say(f"one", cfg.play_sounds, blocking=True)
            time.sleep(0.1)

        save_threads = {}
        i = cfg.start_episode
        while True:
            if i > cfg.num_episodes:
                break
            j = 0
            while j < len(cfg.tasks):
                log_say(f"Recording episode {i} task {cfg.tasks[j]}", cfg.play_sounds)
                dataset = datasets[cfg.tasks[j]]

                # controller.send_stop_tracking()
                # dataset = None
                controller.start(dataset=dataset, episode_time_s=cfg.episode_time_s, single_task=cfg.tasks[j])
                

                if not controller.events["stop_recording"] and (
                    (i < cfg.num_episodes and j == len(cfg.tasks) - 1 and (i % cfg.reset_period == 0)) or controller.events["rerecord_episode"]
                ):
                    reset_time_s = cfg.reset_time_s
                    if (i % cfg.reset_period == 0):
                        reset_time_s = cfg.reset_time_s
                        
                    # reset the environment
                    if reset_time_s > 0:
                        log_say("Reset the environment", cfg.play_sounds, blocking=True)
                        controller.control_loop(episode_time_s=reset_time_s)
                    # reset_environment(client, events, reset_time_s, args.fps)

                if controller.events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    controller.events["rerecord_episode"] = False
                    controller.events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                print("-------------->Saving episode...")
                # if str(j) in save_threads:
                #     save_threads[str(j)].join()
                # save_thread = threading.Thread(target=dataset.save_episode, daemon=True)
                # save_thread.start()
                # save_threads[str(j)] = save_thread
                # save_threads.append(save_thread)



                dataset.save_episode()
                print("-------------->Episode saved.")
                j += 1
            i += 1
            if controller.events["stop_recording"]:
                break
        
        log_say("Reset robot", cfg.play_sounds, blocking=True)
        controller.control_loop(episode_time_s=cfg.reset_time_s)


        # log_say("Saving episodes", cfg.play_sounds, blocking=False)
        # for save_thread in save_threads.values():
        #     save_thread.join()
        # dataset.save_episode()

        # for dataset in datasets.values():
        #     dataset.store_dataset()
        
        log_say("Recording finished", cfg.play_sounds, blocking=False)
        controller.disconnect()

        # if cfg.push_to_hub:
        #     dataset.push_to_hub(tags=cfg.tags, private=cfg.private)