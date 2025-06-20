import time
import threading
from omegaconf import DictConfig, OmegaConf
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from proto.management import BaseCommand, CommandRegistry
from proto.common.utils import get_class
from proto.conf import settings
from proto.common.configs import RecordConfig, ValueName
from proto.common.utils import init_keyboard_listener, log_say


@CommandRegistry.register('record', 'record')
class RecordCommand(BaseCommand):
    help = 'Record sessions'

    def get_features_lerobot_format(self):
        features = {
            "observation.state": {
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

        for module_name, module_data in settings.OBSERVATION.items():
            for sensor_name, (data_name, data_shape) in module_data.items():
                if data_name in [ValueName.POS, ValueName.SPEED]:
                    features[f"observation.state"]["shape"] = tuple(a + b for a, b in zip(features[f"observation.state"]["shape"], data_shape))
                    features[f"observation.state"]["names"].append(sensor_name)
                elif data_name == ValueName.IMAGE:
                    features[f"observation.images.{sensor_name}"] = {
                        "shape": self.cfg.video_res,
                        "names": ["height", "width", "channels"],
                        "info": None,
                        "dtype": "video"
                    }
                    num_cameras += 1
                else:
                    raise ValueError(f"Unknown data type: {data_name}")

        for module_name, module_data in settings.ACTION.items():
            for sensor_name, (data_name, data_shape) in module_data.items():
                if data_name in [ValueName.POS, ValueName.SPEED]:
                    features[f"action"]["shape"] = tuple(a + b for a, b in zip(features[f"action"]["shape"], data_shape))
                    features[f"action"]["names"].append(sensor_name)
                else:
                    raise ValueError(f"Unknown data type: {data_name}")
        return features, num_cameras


    def handle(self, cfg: DictConfig) -> None:
        # print(cfg)
        
        # Instantiate config
        cfg = RecordConfig(**OmegaConf.to_object(cfg))
        self.cfg = cfg

        # Create controller instance
        ControllerClass = get_class(settings.CONTROLLER_CLASS)
        controller = ControllerClass(cfg)
        controller.connect()
        controller.setup()

        features, num_cameras = self.get_features_lerobot_format()

        # Create empty dataset or load existing saved episodes
        # sanity_check_dataset_name(args.repo_id, None)
        datasets = {}
        for task in cfg.tasks:
            dataset = LeRobotDataset.create(
                cfg.repo_id + "_" + task.replace(" ", "_"),
                cfg.fps,
                root='./datasets/' + cfg.repo_id + "_" + task.replace(" ", "_"),
                robot=None,
                robot_type=settings.ROBOT_NAME,
                features=features,
                use_videos=cfg.video,
                image_writer_processes=cfg.num_image_writer_processes,
                image_writer_threads=num_cameras * cfg.num_image_writer_threads_per_camera,
            )
            datasets[task] = dataset

        listener, events = init_keyboard_listener()

        # reset the environment
        log_say("Reset the environment", cfg.play_sounds, blocking=True)
        controller.control_loop(episode_time_s=cfg.reset_time_s)
        log_say(f"Start recording in three", cfg.play_sounds, blocking=True)
        time.sleep(0.1)
        log_say(f"two", cfg.play_sounds, blocking=True)
        time.sleep(0.1)
        log_say(f"one", cfg.play_sounds, blocking=True)
        time.sleep(0.1)

        save_threads = {}
        i = 1
        while True:
            if i > cfg.num_episodes:
                break
            j = 0
            while j < len(cfg.tasks):
                log_say(f"Recording episode {i} task {cfg.tasks[j]}", cfg.play_sounds)
                dataset = datasets[cfg.tasks[j]]
                controller.control_loop(dataset=dataset, events=events, episode_time_s=cfg.episode_time_s, single_task=cfg.tasks[j])

                if not events["stop_recording"] and (
                    (i < cfg.num_episodes and j == len(cfg.tasks) - 1 and (i % cfg.reset_period == 0)) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds, blocking=True)
                    reset_time_s = cfg.reset_time_s
                    if (i % cfg.reset_period == 0):
                        reset_time_s = cfg.reset_time_s
                        
                    # reset the environment
                    controller.control_loop(episode_time_s=reset_time_s)
                    # reset_environment(client, events, reset_time_s, args.fps)

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                print("-------------->Saving episode...")
                # if j in save_threads:
                #     save_threads[j].join()
                # save_thread = threading.Thread(target=dataset.save_episode, daemon=True)
                # save_thread.start()
                # save_threads[j] = save_thread
                # save_threads.append(save_thread)
                dataset.save_episode()
                print("-------------->Episode saved.")
                j += 1
            i += 1
            if events["stop_recording"]:
                break
        
        log_say("Saving episodes", cfg.play_sounds, blocking=False)
        for save_thread in save_threads.values():
            save_thread.join()
        
        controller.disconnect()

        # if cfg.push_to_hub:
        #     dataset.push_to_hub(tags=cfg.tags, private=cfg.private)


        