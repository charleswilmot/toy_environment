import pickle


skin_order = [
    ("Arm1_Left", 0),
    ("Arm2_Left", 0),
    ("Arm2_Left", 1),
    ("Arm2_Left", 2),
    ("Arm1_Left", 2),
    ("Arm1_Right", 0),
    ("Arm2_Right", 0),
    ("Arm2_Right", 1),
    ("Arm2_Right", 2),
    ("Arm1_Right", 2)]
skin_resolution = 10
xlim = [-20.5, 20.5]
ylim = [-13.5, 13.5]
json_model = "../models/two_arms_max_torque_1000_medium_weight_balls.json"
dpi = 10
dt = 1 / 150
n_discrete = 128
env_step_length = 45


args_env = (json_model, skin_order, skin_resolution, xlim, ylim, dpi, env_step_length, dt, n_discrete)

with open("../environments/some_name.pkl", "wb") as f:
    pickle.dump(args_env, f)
