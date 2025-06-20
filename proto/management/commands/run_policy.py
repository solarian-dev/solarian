# from hydra.utils import instantiate
# from omegaconf import DictConfig

# from proto.management import BaseCommand, CommandRegistry
# from proto.utils import log_say, init_keyboard_listener
# from proto.control import control_loop
# from proto.control.lerobot_controller import LeRobotPolicyController


# @CommandRegistry.register('run-policy')
# class RunPolicyCommand(BaseCommand):
#     help = 'Run a policy controller'

#     def handle(self, args: DictConfig) -> None:
#         # Instantiate robot config with Hydra
#         cfg = instantiate(args.proto)
#         if not cfg.policy_path:
#             raise ValueError("policy_path is required in configuration")

#         # Initialize keyboard listener
#         init_keyboard_listener()

#         # Policy config from Hydra
#         client = LeRobotPolicyController(
#             relay_server_ip=cfg.server.ip,
#             policy_path=cfg.policy_path,
#             robot_config=cfg,
#             fps=cfg.get('fps', 60),
#             play_sounds=cfg.get('play_sounds', True),
#             display_cameras=cfg.get('display_cameras', False),
#             single_task=cfg.get('single_task', None),
#             repo_id=cfg.get('repo_id', None)
#         )

#         try:
#             client.connect()
#             log_say("Connected to relay server")
#             control_loop(client)
#         except KeyboardInterrupt:
#             print("\nShutting down policy controller...")
#         finally:
#             client.disconnect()

#         log_say(f"Starting policy", cfg.get('play_sounds', True), blocking=True)






    

    
