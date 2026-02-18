# import torch
# import time

# from .base import Controller

# class LeRobotPolicyController(Controller):
#     def __init__(self, relay_server_ip, robot_config):
#         super().__init__(relay_server_ip, robot_config)

#         self.policy = None


#     def load_policy(self, policy_path, config):
#         from lerobot.common.policies.act.modeling_act import ACTPolicy
#         self.policy = ACTPolicy.from_pretrained(policy_path, config=config)
#         # from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
#         # self.policy = PI0FASTPolicy.from_pretrained(policy_path, config=config)
#         device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#         self.policy.to(device)
        
#     def reset(self):
#         self.policy.reset()
#         message = {
#             "start_pos": {}
#         }
#         self.start_pos = True
#         self.send_action(message)

#     def control_step(self, record_data: bool = False) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
#         device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#         self.policy.eval()
#         with torch.no_grad():
#             start_time = time.perf_counter()

#             # Read the follower state and access the frames from the cameras
#             observation = self.capture_observation()


#             # print("observation:", observation)
#             # print("observation.state type:", type(observation["observation.state"]))
#             # print("observation.state shape:", observation["observation.state"].shape)
#             # print("observation.state dtype:", observation["observation.state"].dtype)
            
#             # print("observation.images.front type:", type(observation["observation.images.front"]))
#             # print("observation.images.front shape:", observation["observation.images.front"].shape)
#             # print("observation.images.front dtype:", observation["observation.images.front"].dtype)
            
#             # print("observation.images.wrist type:", type(observation["observation.images.wrist"]))
#             # print("observation.images.wrist shape:", observation["observation.images.wrist"].shape)
#             # print("observation.images.wrist dtype:", observation["observation.images.wrist"].dtype)


#             # Convert to pytorch format: channel first and float32 in [0,1]
#             # with batch dimension
#             for name in observation:
#                 if "image" in name:
#                     observation[name] = observation[name].type(torch.float32) / 255
#                     observation[name] = observation[name].permute(2, 0, 1).contiguous()
#                 observation[name] = observation[name].unsqueeze(0)
#                 observation[name] = observation[name].to(device)

#             observation['task'] = "Grasp a lego block and put it in the bin."
            
#             # print("Observation type", type(observation))
#             # print("Observation", observation)

#             # Compute the next action with the policy
#             # based on the current observation
#             action = self.policy.select_action(observation)

#             print("Action value", action)
#             print("Action type", type(action))
#             print("Action shape", action.shape)
#             print("Action dtype", action.dtype)
            
            
#             # Remove batch dimension
#             action = action.squeeze(0)
#             # Move to cpu, if not already the case
#             action = action.to("cpu")

#             print("Action value", action)
#             print("Action type", type(action))
#             print("Action shape", action.shape)
#             print("Action dtype", action.dtype)
            
#             # Order the robot to move
#             self.send_action(action)

#             dt_s = time.perf_counter() - start_time

    

    