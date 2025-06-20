import json


with open(".cache/calibration/lekiwi2/lekiwi_leader_arm3.json", "r") as f:
    leader_calibration = json.load(f)


with open(".cache/calibration/lekiwi2/lekiwi_follower_arm.json", "r") as f:
    follower_calibration = json.load(f)
    
for joint_name in leader_calibration.keys():
    if joint_name == "gripper":
        continue
    HOME = 2047
    leader_calibration[joint_name]["range_min"] = follower_calibration[joint_name]["range_min"]
    leader_calibration[joint_name]["range_max"] = follower_calibration[joint_name]["range_max"]
    
with open(".cache/calibration/lekiwi2/lekiwi_leader_arm_translated.json", "w") as f:
    json.dump(leader_calibration, f, indent=4)
    
    