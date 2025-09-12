import torch
import os
import math
import sys
from pathlib import Path
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig

from proto.management import BaseCommand, CommandRegistry
from proto.common.configs import TrainConfig
from proto.analytics import LoggerFactory


import os
import time
import random
from typing import Optional, Tuple, Dict, Any

import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler


def save_training_state(
    path: str,
    optimizer: Optimizer,
    *,
    step: int,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    save_rng_state: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save training state for resume:
      - optimizer state
      - learning-rate scheduler state (if any)
      - AMP GradScaler state (if any)
      - global step (int)
      - RNG states (optional: torch, cuda, numpy, random)
      - any extra metadata you want (dict)

    Notes:
    - This does NOT save model weights (use model.state_dict() separately).
    - File is written atomically by saving to a temp file then renaming.
    """
    state = {
        "version": 1,
        "timestamp": time.time(),
        "step": int(step),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }

    if save_rng_state:
        state["rng"] = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    # Atomic write
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_training_state(
    path: str,
    optimizer: Optimizer,
    *,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    map_location: str = "cpu",
    restore_rng_state: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    """
    Load training state and restore into the provided optimizer/scheduler/scaler.

    Returns:
      step (int), extra (dict)

    Usage pattern:
      step, extra = load_training_state(ckpt, optimizer, scheduler=my_sched, scaler=my_scaler)
      # resume your loop from `step+1`
    """
    state = torch.load(path, map_location=map_location)

    # Optimizer
    optimizer.load_state_dict(state["optimizer"])

    # Scheduler (if provided & present)
    if scheduler is not None and state.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(state["scheduler"])
        except Exception as e:
            # In case of scheduler type/param mismatch, don't hard-crash.
            print(f"[load_training_state] Warning: could not load scheduler state ({e}).")

    # GradScaler (if provided & present)
    if scaler is not None and state.get("scaler") is not None:
        try:
            scaler.load_state_dict(state["scaler"])
        except Exception as e:
            print(f"[load_training_state] Warning: could not load GradScaler state ({e}).")

    # RNG
    if restore_rng_state and state.get("rng"):
        rng = state["rng"]
        try:
            random.setstate(rng["python_random"])
            np.random.set_state(rng["numpy_random"])
            torch.set_rng_state(rng["torch_cpu"])
            if torch.cuda.is_available() and rng["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
        except Exception as e:
            print(f"[load_training_state] Warning: could not restore RNG state ({e}).")

    step = int(state.get("step", 0))
    extra = state.get("extra", {})
    return step, extra



@CommandRegistry.register('train', 'train')
class TrainCommand(BaseCommand):
    help = 'Train a model.'
    
    def get_config_class(self):
        """Return the config class for this command."""
        return TrainConfig

    def handle(self, cfg: TrainConfig) -> None:
        try:
            # Create a directory to store the training checkpoint.
            if cfg.output_dir is not None:
                output_directory = Path(cfg.output_dir)
            else:
                output_directory = Path(os.getenv("PROTO_PROJECT_PATH") + "/models/" + cfg.job_name)
            output_directory.mkdir(parents=True, exist_ok=True)
            cfg.output_dir = str(output_directory)

            # Create logger if enabled
            logger = None
            if cfg.logger.type != 'none':
                logger = LoggerFactory.create_logger(cfg.logger, train_cfg=cfg)

            # Select your device
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

            # Number of offline training steps (we'll only do offline training for this example.)
            # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
            training_steps = cfg.steps
            log_freq = cfg.log_freq

            # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
            # creating the policy:
            #   - input/output shapes: to properly size the policy
            #   - dataset stats: for normalization and denormalization of input/outputs


            if cfg.train_dataset.startswith("local@"):
                train_repo_id = cfg.train_dataset.split("@")[1]
                train_root = os.getenv("PROTO_PROJECT_PATH") + "/datasets/" + train_repo_id
                train_revision = '-'
            else:
                train_repo_id = cfg.train_dataset
                train_root = None
                train_revision = None

            print(f"Dataset: {train_repo_id}")
            # print(f"Selected root: {train_root}")

            train_ds_meta = LeRobotDatasetMetadata(repo_id=train_repo_id, root=train_root, revision=train_revision)
            features = dataset_to_policy_features(train_ds_meta.features)
            output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
            input_features = {key: ft for key, ft in features.items() if key not in output_features}

            print(f"Total episodes: {train_ds_meta.total_episodes}")
            print(f"Avg frames per episode: {train_ds_meta.total_frames / train_ds_meta.total_episodes:.3f}")
            print(f"FPS: {train_ds_meta.fps}")
            # print(f"Robot type: {train_ds_meta.robot_type}")
            # print(f"keys to access images from cameras: {train_ds_meta.camera_keys=}\n")
            print("Chunk size: ", train_ds_meta.chunks_size)
            print("\n")
            # print("Features: ", train_ds_meta.features)
            # print("\n")
            print("Input features: ", input_features)
            print("\n")
            print("Output features: ", output_features)
            print("\n")


            # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
            # we'll just use the defaults and so no arguments other than input/output features need to be passed.
            # cfg = DiffusionConfig(input_features=input_features, output_features=output_features)

            # # We can now instantiate our policy with this config and the dataset stats.
            # policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
            # policy.train()

            config = ACTConfig(
                input_features=input_features,
                output_features=output_features,
                n_obs_steps=cfg.model.n_obs_steps,
                chunk_size=cfg.model.chunk_size,
                n_action_steps=cfg.model.n_action_steps,
                use_vae=cfg.model.use_vae,
                temporal_ensemble_coeff=cfg.model.temporal_ensemble_coeff
            )

            if cfg.resume:
                policy = ACTPolicy.from_pretrained(cfg.policy_path, config=config)
            else:
                policy = ACTPolicy(config=config, dataset_stats=train_ds_meta.stats)
            
            policy.train()
            policy.to(device)

            delta_timestamps = {
                # "observation.image": [i / ds_meta.fps for i in cfg.observation_delta_indices],
                # "observation.state": [i / ds_meta.fps for i in config.observation_delta_indices],
                # "observation.environment_state": [i / ds_meta.fps for i in config.observation_delta_indices],
                "action": [i / train_ds_meta.fps for i in config.action_delta_indices],
            }

            if cfg.val_percentage is not None:
                # - Calculate train and val episodes
                total_episodes = train_ds_meta.total_episodes
                episodes = list(range(train_ds_meta.total_episodes))
                num_train_episodes = math.floor(total_episodes * (1 - cfg.val_percentage))
                num_train_frames = train_ds_meta.total_frames * (1 - cfg.val_percentage)
                train_episodes = episodes[:num_train_episodes]
                val_episodes = episodes[num_train_episodes:]
                num_val_episodes = len(val_episodes)
                num_val_frames = train_ds_meta.total_frames - num_train_frames
                
                # - Load train and val datasets
                train_dataset = LeRobotDataset(train_repo_id, root=train_root, revision=train_revision, episodes=train_episodes, delta_timestamps=delta_timestamps)
                
                # Create a new dataset with remapped episode indices for validation
                # This is necessary because LeRobotDataset expects episode indices to be 0-based
                # when accessing them through episode_data_index
                val_dataset = LeRobotDataset(train_repo_id, root=train_root, revision=train_revision, episodes=val_episodes, delta_timestamps=delta_timestamps)
                
            else:
                num_train_episodes = train_ds_meta.total_episodes
                num_train_frames = train_ds_meta.total_frames
                num_val_episodes = 0
                num_val_frames = 0
                
                train_dataset = LeRobotDataset(train_repo_id, root=train_root, revision=train_revision, delta_timestamps=delta_timestamps)
                if cfg.val_dataset is not None:
                    val_repo_id = cfg.val_dataset.split("@")[1]
                    val_root = os.getenv("PROTO_PROJECT_PATH") + "/datasets/" + val_repo_id
                    val_revision = '-'
                    val_dataset = LeRobotDataset(val_repo_id, root=val_root, revision=val_revision, delta_timestamps=delta_timestamps)
                    val_ds_meta = LeRobotDatasetMetadata(repo_id=val_repo_id, root=val_root, revision=val_revision)
                    num_val_episodes = val_ds_meta.total_episodes
                    num_val_frames = val_ds_meta.total_frames
                else:
                    val_dataset = None

            val_batch_size = min(cfg.batch_size, len(val_dataset))
            
            # Create dataloader for evaluation.
            if val_dataset is not None:
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    num_workers=cfg.num_workers,
                    batch_size=val_batch_size,
                    shuffle=False,
                    pin_memory=(device == torch.device("cuda")),
                    drop_last=False,
                )
                
            # Then we create our optimizer and dataloader for offline training.
            # TODO: support other optimizers use a factory pattern
            optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.optimizer.lr)
            
            if cfg.scheduler.type is not None:
                # TODO: support other schedulers use a factory pattern
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            else:
                lr_scheduler = None

            if cfg.precision.amp_enabled:
                grad_scaler = torch.amp.GradScaler(
                    init_scale=cfg.precision.init_scale, 
                    growth_interval=cfg.precision.growth_interval, 
                    growth_factor=cfg.precision.growth_factor
                )
            else:
                grad_scaler = None
            
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                num_workers=cfg.num_workers,
                batch_size=cfg.batch_size,
                shuffle=True,
                pin_memory=(device == torch.device("cuda")),
                drop_last=True,
            )

            num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            num_total_params = sum(p.numel() for p in policy.parameters())
            print(f"Number of learnable parameters: {num_learnable_params}")
            print(f"Number of total parameters: {num_total_params}")
            print(f"Number of frames in training dataset: {num_train_frames}")
            print(f"Number of episodes in training dataset: {num_train_episodes}")
            print(f"Number of frames in validation dataset: {num_val_frames}")
            print(f"Number of episodes in validation dataset: {num_val_episodes}")

            if cfg.resume:
                # load training state
                step, extra = load_training_state(
                    cfg.policy_path + "/train_state.pt", 
                    optimizer, 
                    scheduler=lr_scheduler, 
                    scaler=grad_scaler
                )
            else:
                step = 0

            # Run training loop.
            done = False
            training_loss = 0
            average_loss = 0
            epoch = 0
            samples = 0
            with tqdm(total=training_steps, desc="Training") as pbar:
                while not done:
                    
                    for batch in train_dataloader:
                        # iterate over training batches

                        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                        
                        if cfg.precision.amp_enabled:
                            with torch.autocast():
                                loss, loss_dict = policy.forward(batch)
                        else:
                            loss, loss_dict = policy.forward(batch)
                        
                        
                        if cfg.precision.amp_enabled:
                            grad_scaler.scale(loss).backward()
                            grad_scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                            grad_scaler.step(optimizer)
                            grad_scaler.update()
                        else:
                            loss.backward()
                            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                            optimizer.step()

                        optimizer.zero_grad()

                        if lr_scheduler is not None:
                            lr_scheduler.step()

                        if hasattr(policy, "update") and callable(getattr(policy, "update")):
                            policy.update()

                        step += 1
                        samples += cfg.batch_size
                        episodes = samples / (num_train_frames / num_train_episodes)
                        epoch = samples / num_train_frames

                        training_loss = loss.item()
                        pbar.set_postfix({"epoch": epoch, "train_loss": training_loss})
                        pbar.update(1)
                        log_data = {}

                        if step % log_freq == 0:
                            training_loss = loss.item()
                            # print(f"step: {step} loss: {loss.item():.3f}")
                            if logger is not None:
                                from proto.analytics.train_logger import LogTrainData
                                log_data.update({
                                    "train/loss": training_loss,
                                    "train/lr": optimizer.param_groups[0]["lr"],
                                    "train/epoch": epoch,
                                    "train/episodes": episodes,
                                    "train/samples": samples,
                                })
                                for key, value in loss_dict.items():
                                    log_data[f"train/{key}"] = value
                                logger.log(
                                    LogTrainData(
                                        step=step, 
                                        data=log_data
                                    )
                                )
                                
                        # save checkpoint
                        if step % cfg.save_freq == 0:
                            step_str = str(step).zfill(6)
                            # save model params checkpoint
                            policy.save_pretrained(output_directory / f"checkpoints/{step_str}/pretrained_model")
                            # save training state checkpoint
                            save_training_state(
                                path=output_directory / f"checkpoints/{step_str}/pretrained_model/train_state.pt",
                                optimizer=optimizer,
                                step=step,
                                scheduler=lr_scheduler,
                                scaler=grad_scaler,
                            )

                            # perform evaluation on validation dataset
                            if val_dataset is not None:
                                policy.eval()
                                with (
                                    torch.inference_mode(),
                                    torch.autocast(device_type=device.type, enabled=cfg.precision.amp_enabled)
                                ):
                                    loss_cumsum = 0
                                    j = 0
                                    average_loss = 0
                                    pbar_val = tqdm(val_dataloader, desc=f"Validation at step: {step}")
                                    for eval_batch in pbar_val:
                                        # print(f"eval_batch: {j}")
                                        j += 1
                                        eval_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in eval_batch.items()}
                                        
                                        eval_loss, _ = policy.forward(eval_batch)
                                        
                                        # loss.backward()
                                        # optimizer.step()
                                        # optimizer.zero_grad()
                                        
                                        loss_cumsum += eval_loss.item()
                                        average_loss = loss_cumsum / j
                                        pbar_val.set_postfix({"val_loss": average_loss, "train_loss": training_loss})

                                # Calculate the average loss over the validation set.
                                # average_loss = loss_cumsum / len(val_dataloader)
                                # print(f"Validation loss: {average_loss:.3f}")
                                if logger is not None:
                                    from proto.analytics.train_logger import LogTrainData
                                    log_data.update({
                                        "validate/loss": average_loss,
                                    })
                                    for key, value in loss_dict.items():
                                        log_data[f"validate/{key}"] = value
                                    logger.log(
                                        LogTrainData(
                                            step=step, 
                                            data=log_data
                                        )
                                    )
                                # reset policy to training mode
                                policy.train()
                        
                        if step >= training_steps:
                            done = True
                            break

                        
        except Exception as e:
            # Handle exceptions with clear error messages
            print(f"\nError during training: {str(e)}")
            import traceback
            print(traceback.format_exc())
            sys.exit(1)
