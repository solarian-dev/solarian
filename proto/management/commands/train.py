import torch
import os
import math
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from proto.management import BaseCommand, CommandRegistry
from proto.common.configs import TrainConfig
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig


@CommandRegistry.register('train', 'train')
class TrainCommand(BaseCommand):
    help = 'Train a model.'

    def handle(self, cfg: TrainConfig) -> None:

        # Create a directory to store the training checkpoint.
        output_directory = Path(os.getenv("PROTO_PROJECT_PATH") + "/models/" + cfg.job_name)
        output_directory.mkdir(parents=True, exist_ok=True)

        # # Select your device
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

        print(f"Selected dataset: {train_repo_id}")
        print(f"Selected root: {train_root}")

        train_ds_meta = LeRobotDatasetMetadata(repo_id=train_repo_id, root=train_root, revision=train_revision)
        features = dataset_to_policy_features(train_ds_meta.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}

        print(f"Total number of episodes: {train_ds_meta.total_episodes}")
        print(f"Average number of frames per episode: {train_ds_meta.total_frames / train_ds_meta.total_episodes:.3f}")
        print(f"Frames per second used during data collection: {train_ds_meta.fps}")
        print(f"Robot type: {train_ds_meta.robot_type}")
        print(f"keys to access images from cameras: {train_ds_meta.camera_keys=}\n")
        print("Chunk size: ", train_ds_meta.chunks_size)
        print("\n")
        print("Features: ", train_ds_meta.features)
        print("\n")
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
            train_episodes = episodes[:num_train_episodes]
            val_episodes = episodes[num_train_episodes:]
            print(f"Number of episodes in training dataset: {len(train_episodes)}")
            print(f"Number of episodes in validation dataset: {len(val_episodes)}")
            
            # - Load train and val datasets
            train_dataset = LeRobotDataset(train_repo_id, root=train_root, revision=train_revision, episodes=train_episodes, delta_timestamps=delta_timestamps)
            
            # Create a new dataset with remapped episode indices for validation
            # This is necessary because LeRobotDataset expects episode indices to be 0-based
            # when accessing them through episode_data_index
            val_dataset = LeRobotDataset(train_repo_id, root=train_root, revision=train_revision, episodes=val_episodes, delta_timestamps=delta_timestamps)
            
        else:
            print(f"Number of episodes in training dataset: {train_ds_meta.total_episodes}")
            train_dataset = LeRobotDataset(train_repo_id, root=train_root, revision=train_revision, delta_timestamps=delta_timestamps)
            if cfg.val_dataset is not None:
                val_repo_id = cfg.val_dataset.split("@")[1]
                val_root = os.getenv("PROTO_PROJECT_PATH") + "/datasets/" + val_repo_id
                val_revision = '-'
                val_dataset = LeRobotDataset(val_repo_id, root=val_root, revision=val_revision, delta_timestamps=delta_timestamps)
                val_ds_meta = LeRobotDatasetMetadata(repo_id=val_repo_id, root=val_root, revision=val_revision)
                print(f"Number of episodes in validation dataset: {val_ds_meta.total_episodes}")
                
            else:
                val_dataset = None

        print(f"Number of frames in training dataset: {len(train_dataset)}")


        val_batch_size = min(cfg.batch_size, len(val_dataset))
        print(f"Validation batch size: {val_batch_size}")
        

        # Create dataloader for evaluation.
        if val_dataset is not None:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                num_workers=cfg.num_workers,
                batch_size=val_batch_size,
                shuffle=False,
                pin_memory=device != torch.device("cpu"),
                drop_last=False,
            )
            print(f"Number of frames in validation dataset: {len(val_dataset)}")



        # Then we create our optimizer and dataloader for offline training.
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=device != torch.device("cpu"),
            drop_last=True,
        )

        

        # Run training loop.
        step = 1
        done = False
        while not done:
            for batch in train_dataloader:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                
                loss, _ = policy.forward(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if step % log_freq == 0:
                    print(f"step: {step} loss: {loss.item():.3f}")
                    # print(f"step: {step}")

                # save checkpoint
                if step % cfg.save_freq == 0:
                    step_str = str(step).zfill(6)
                    policy.save_pretrained(output_directory / f"checkpoints/{step_str}/pretrained_model")

                    # perform evaluation on validation dataset
                    if val_dataset is not None:
                        policy.eval()
                        with torch.inference_mode():
                            loss_cumsum = 0
                            j = 0
                            for eval_batch in val_dataloader:
                                print(f"eval_batch: {j}")
                                j += 1
                                eval_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in eval_batch.items()}
                                
                                eval_loss, _ = policy.forward(eval_batch)
                                
                                # loss.backward()
                                # optimizer.step()
                                # optimizer.zero_grad()
                                
                                loss_cumsum += eval_loss.item()

                        # Calculate the average loss over the validation set.
                        average_loss = loss_cumsum / len(val_dataloader)
                        print(f"Validation loss: {average_loss:.3f}")
                        # reset policy to training mode
                        policy.train()

                step += 1
                if step >= training_steps:
                    done = True
                    break

        

        
