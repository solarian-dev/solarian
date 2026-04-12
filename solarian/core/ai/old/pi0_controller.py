# import time
# import torch
# import cv2

# from openpi_client import websocket_client_policy
# from openpi_client import image_tools

# from .base import Controller
# from solarian.utils import busy_wait


# class PI0Controller(Controller):
#     def __init__(self, relay_server_ip, robot_config, pi0_host, pi0_port):
#         super().__init__(relay_server_ip, robot_config)

#         self.pi0_client = websocket_client_policy.WebsocketClientPolicy(host=pi0_host, port=pi0_port)


#     def reset(self):
#         message = {
#             "start_pos": {}
#         }
#         self.start_pos = True
#         self.send_action(message)

#     def control_step(self, record_data: bool = False) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

#         observation = self.capture_observation()
#         observation_pi0 = {}
#         fps = 30

#         observation_pi0 = {}
        
#         for name in observation:
#             key = name.replace("observation.", "observation/")
            
#             if "image" in name:
#                 # observation_pi0[key] = image_tools.resize_with_pad(observation[name].numpy()[np.newaxis, :], 224, 224)
#                 observation_pi0[key] = image_tools.resize_with_pad(observation[name].numpy(), 224, 224)
#                 print(f"{key} center value:" , observation_pi0[key][112,112])

#                 # show image
#                 cv2.imshow(key, cv2.cvtColor(observation_pi0[key], cv2.COLOR_RGB2BGR))
#                 cv2.waitKey(1)
#             else:
#                 observation_pi0[key] = observation[name].numpy()
#                 print(f"{key} value:" , observation_pi0[key])
            
#             print(f"{key} type:" , type(observation_pi0[key]))
#             print(f"{key} shape:" , observation_pi0[key].shape)
#             print(f"{key} dtype:" , observation_pi0[key].dtype)
        
#         observation_pi0['prompt'] = "Pick up the lego block."
        
#         start_time = time.perf_counter()
#         action_chunk = self.pi0_client.infer(observation_pi0)["actions"]
#         dt_s = time.perf_counter() - start_time
#         print(f"Policy step time: {dt_s:.3f} seconds")
       
       
#         for i in range(10):
#             #range(action_chunk.shape[0]):
#             start_loop_t = time.perf_counter()
#             # print(f"action {i}: {action_chunk[i]}")
#             t = torch.tensor(action_chunk[i]).float()
#             print(f"action {i} tensor:", t)
#             # Order the robot to move
#             self.send_action(t)
#             dt_s = time.perf_counter() - start_loop_t
#             busy_wait(1 / fps - dt_s)