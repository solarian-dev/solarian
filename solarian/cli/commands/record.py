import logging
import time

from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.configs import RecordConfig
from solarian.utils import StatusBoard, get_class, log_say, setup_process_logging

logger = logging.getLogger(__name__)

#: How long to give the robot to report in before deciding it is not recording.
_ROBOT_REPORT_TIMEOUT_S = 5.0


@CommandRegistry.register('record', 'record')
class RecordCommand(BaseCommand):
    help = "Teleoperate and record a training dataset, episode by episode"
    description = """
    The same teleop loop as `run-controller`, cut into episodes: reset the scene,
    record for episode_time_s, seal, repeat. The robot is what writes the dataset
    — one mp4 per camera at archive resolution plus one parquet of every signal at
    its native rate — so `conf/robot.yaml`'s `recording:` block decides where and
    how, and `conf/record.yaml` decides only the shape of the session.

    The robot (`solarian run-robot`) and the relay (`solarian run-server`) have to
    be up. Press delete during an episode to throw it away and record it again: the
    index is reused, so a bad take leaves no gap in the dataset. Press r to seal
    early, and quit the UI to end the session.
    """
    examples = (
        "solarian record",
        "solarian record num_episodes=50 episode_time_s=20",
        "solarian record tasks='[put the cup on the saucer]'",
    )

    def get_config_class(self):
        return RecordConfig

    def handle(self, cfg: DictConfig) -> None:
        from solarian.conf import settings

        log_level = getattr(logging, getattr(settings, 'LOG_LEVEL', 'INFO'), logging.INFO)
        setup_process_logging(level=log_level, logger_name=None)

        cfg = RecordConfig(**OmegaConf.to_object(cfg))
        # The schedule below owns the episode boundaries, so the controller
        # must not also open one of its own.
        cfg.record = False

        controller = get_class(settings.CONTROLLER_CLASS)(cfg)

        try:
            controller.connect()
            controller.initialize()
            controller.start(session=lambda: self.record_session(controller, cfg))
        except KeyboardInterrupt:
            StatusBoard.stop()
        except Exception:
            StatusBoard.print_exception()
        finally:
            controller.disconnect()

    def record_session(self, controller, cfg: RecordConfig) -> None:
        """Bracket ``num_episodes`` takes, alongside the controller's own loop.

        The controller keeps teleoperating throughout — between episodes as
        much as during them — so the operator resets the scene by hand and
        nothing is torn down and rebuilt between takes.
        """
        # The robot writes the dataset, so it is the only thing that can say
        # whether one will exist. Finding out after fifty takes is too late.
        deadline = time.monotonic() + _ROBOT_REPORT_TIMEOUT_S
        while not controller.recording_state["enabled"] and time.monotonic() < deadline:
            time.sleep(0.05)
        if not controller.recording_state["enabled"]:
            raise RuntimeError(
                "The robot is not recording: either it never reported in, or "
                "conf/robot.yaml sets recording.enabled: false"
            )

        tasks = cfg.tasks or [cfg.task]
        self.reset(controller, cfg)

        episode = 0
        while episode < cfg.num_episodes and controller.running:
            log_say(f"Recording episode {episode + 1} of {cfg.num_episodes}",
                    cfg.play_sounds)
            controller.start_recording(tasks[episode % len(tasks)], session=cfg.name)
            # Returns early when the operator seals with r or discards with x.
            controller.episode_over.wait(timeout=cfg.episode_time_s)
            controller.stop_recording()

            if controller.events["rerecord_episode"]:
                controller.events["rerecord_episode"] = False
                log_say("Re-recording", cfg.play_sounds)
                self.reset(controller, cfg)
                continue

            episode += 1
            if episode < cfg.num_episodes and episode % cfg.reset_period == 0:
                self.reset(controller, cfg)

        log_say(f"Recorded {episode} episodes", cfg.play_sounds, blocking=True)

    def reset(self, controller, cfg: RecordConfig) -> None:
        """Give the operator reset_time_s to set the scene up, still teleoperating."""
        if cfg.reset_time_s <= 0:
            return
        log_say("Reset the environment", cfg.play_sounds)
        deadline = time.monotonic() + cfg.reset_time_s
        while controller.running and time.monotonic() < deadline:
            time.sleep(0.05)
