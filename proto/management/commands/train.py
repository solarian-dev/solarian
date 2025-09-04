import torch
import os
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
            repo_id = cfg.train_dataset.split("@")[1]
            root = os.getenv("PROTO_PROJECT_PATH") + "/datasets/" + repo_id
            revision = '-'
        else:
            repo_id = cfg.train_dataset
            root = None
            revision = None

        print(f"Selected dataset: {repo_id}")
        print(f"Selected root: {root}")

        ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root, revision=revision)
        features = dataset_to_policy_features(ds_meta.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}

        print(f"Total number of episodes: {ds_meta.total_episodes}")
        print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
        print(f"Frames per second used during data collection: {ds_meta.fps}")
        print(f"Robot type: {ds_meta.robot_type}")
        print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")
        print("Chunk size: ", ds_meta.chunks_size)
        print("\n")
        print("Features: ", ds_meta.features)
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
            policy = ACTPolicy(config=config, dataset_stats=ds_meta.stats)
        
        policy.train()
        policy.to(device)

        delta_timestamps = {
            # "observation.image": [i / ds_meta.fps for i in cfg.observation_delta_indices],
            # "observation.state": [i / ds_meta.fps for i in config.observation_delta_indices],
            # "observation.environment_state": [i / ds_meta.fps for i in config.observation_delta_indices],
            "action": [i / ds_meta.fps for i in config.action_delta_indices],
        }

        dataset = LeRobotDataset(repo_id=repo_id, root=root, revision=revision, delta_timestamps=delta_timestamps)

        # And see how many frames you have:
        print(f"Selected episodes: {dataset.episodes}")
        print(f"Number of episodes selected: {dataset.num_episodes}")
        print(f"Number of frames selected: {dataset.num_frames}")

        # Then we create our optimizer and dataloader for offline training.
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=device.type != "cpu",
            drop_last=True,
        )

        # Run training loop.
        step = 0
        done = False
        while not done:
            for batch in dataloader:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                
                loss, _ = policy.forward(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if step % log_freq == 0:
                    print(f"step: {step} loss: {loss.item():.3f}")

                # save checkpoint
                if step % cfg.save_freq == 0:
                    step_str = str(step).zfill(6)
                    policy.save_pretrained(output_directory / f"checkpoints/{step_str}/pretrained_model")

                    # perform evaluation on validation dataset
                    # TODO

                step += 1
                if step >= training_steps:
                    done = True
                    break

        # Save a policy checkpoint.
        policy.save_pretrained(output_directory)

        
