import sys
sys.path.insert(0, '../src/')
import environment
import viewer
import matplotlib.pyplot as plt
import pickle
import numpy as np


def actions_dict_from_array(actions):
    return {
        "Arm1_to_Arm2_Left": actions[0],
        "Arm1_to_Arm2_Right": actions[1],
        "Ground_to_Arm1_Left": actions[2],
        "Ground_to_Arm1_Right": actions[3]
    }


env_conf_path = "../environments/some_name.pkl"
with open(env_conf_path, "rb") as f:
    env = environment.Environment(*pickle.load(f))

window = viewer.SimpleJointAgentWindow()
for i in range(1000):
    print("Step {: 5d}".format(i), end="\r")
    pos = np.random.uniform(size=4, low=-1, high=1) * [2.8, 2.8, 3.14, 3.14]
    env.set_positions(actions_dict_from_array(pos))
    env.env_step()
    vision = env.vision
    positions = env.discrete_positions
    tactile = env.tactile
    window(vision, positions, tactile)
