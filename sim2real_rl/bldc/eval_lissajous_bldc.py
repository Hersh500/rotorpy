import os
from datetime import datetime
import torch
import roma
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import rotorpy
from rotorpy.vehicles.crazyfliebrushless_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorDiffTrackingEnv
from rotorpy.learning.quadrotor_reward_functions import vec_diff_reward_negative
from rotorpy.learning.learning_utils import *
from rotorpy.trajectories.hover_traj import BatchedHoverTraj
from rotorpy.trajectories.lissajous_traj import BatchedTwoDLissajous
from rotorpy.controllers.quadrotor_control import BatchedSE3Control
from rotorpy.world import World


# Next import Stable Baselines.
try:
    import stable_baselines3
except:
    raise ImportError('To run this example you must have Stable Baselines installed via pip install stable_baselines3')

from stable_baselines3 import PPO                                   # We'll use PPO for training.
from stable_baselines3.ppo.policies import MlpPolicy                # The policy will be represented by an MLP
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

device = torch.device("cpu")

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "learning", "policies", "PPO",
                         "lissajous_bldc_cmd_ctattMay-29-18-40")
model_file = "liss_bldc_5996544_steps"

# model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "learning", "policies", "PPO",
#                          "lissajous_bldc_cmd_ctatt_noaero_May-29-20-02")
# model_file = "liss_bldc_5996544_steps"
model = PPO.load(os.path.join(model_dir, model_file))

action_history_length = 3
pos_history_length = 3
lookahead_length = 5
init_rotor_speed = 1676.57
aero = True

reward_weights = {'x': 1.0, 
                  'v': 0.3, 
                  'yaw': 0.3,
                  'w': 2e-3, 
                  'u': np.array([5e-2, 5e-2, 5e-2, 5e-2]), 
                  'u_mag': np.array([0e-4, 0e-4, 0e-4, 0e-4]), 
                  'survive': 5}

reward_fn = lambda obs, action, **kwargs: vec_diff_reward_negative(obs, action, reward_weights, **kwargs)

control_mode = "cmd_ctatt"

wbound = 10
world = World.empty((-wbound, wbound, -wbound,
                     wbound, -wbound, wbound))

# Create eval environment - set up initial states and trajectory for eval. These could be different from the training env.
num_eval_envs = 5

eval_reset_options = dict(rotorpy.learning.quadrotor_environments.DEFAULT_RESET_OPTIONS)
eval_reset_options["params"] = "fixed"
eval_reset_options["initial_states"] = "deterministic"
eval_reset_options["pos_bound"] = 2.0
eval_reset_options["vel_bound"] = 0.2
eval_reset_options["traj_randomization_fn"] = None

A_s = torch.ones(num_eval_envs, device=device) * 1
B_s = torch.ones(num_eval_envs, device=device) * 4
a_s = torch.ones(num_eval_envs, device=device) * 2.0
b_s = torch.ones(num_eval_envs, device=device) * 0.8
delta_s = torch.zeros(num_eval_envs, device=device)
x_offset_s = torch.zeros(num_eval_envs, device=device)
y_offset_s = torch.zeros(num_eval_envs, device=device)
height_s = torch.zeros(num_eval_envs, device=device)
yaw_bool_s = torch.zeros(num_eval_envs, device=device)

eval_trajectory = BatchedTwoDLissajous(A_s, B_s, a_s, b_s, delta_s, x_offset_s, y_offset_s, height_s, yaw_bool_s, device=device)

print(f"DOING FINAL EVALUATION...")
init_rotor_speed = 1676.57

# generate random initial conditions
# x0 = {'x': torch.rand(num_eval_envs,3, device=device).double() * 4 - 2,
#         'v': torch.rand(num_eval_envs, 3, device=device).double() * 0.1, 
#         'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_eval_envs, 1).double(),
#         'w': torch.zeros(num_eval_envs, 3, device=device).double(),
#         'wind': torch.zeros(num_eval_envs, 3, device=device).double(),
#         'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_eval_envs, 1).double()}

x0 = {'x': torch.ones(num_eval_envs,3, device=device).double(),
        'v': torch.zeros(num_eval_envs, 3, device=device).double(),
        'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_eval_envs, 1).double(),
        'w': torch.zeros(num_eval_envs, 3, device=device).double(),
        'wind': torch.zeros(num_eval_envs, 3, device=device).double(),
        'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_eval_envs, 1).double()}


env_for_policy = QuadrotorDiffTrackingEnv(num_eval_envs, 
                              initial_states=x0, 
                              trajectory=eval_trajectory,
                              quad_params=dict(quad_params), 
                              max_time=5, 
                              world=world,
                              control_mode=control_mode, 
                              device=device,
                              render_mode="3D",
                              reward_fn=reward_fn,
                              reset_options=eval_reset_options,
                              action_history_length=action_history_length,
                              pos_history_length=pos_history_length,
                              trace_dynamics=True,
                              traj_lookahead_length=lookahead_length,
                              aero=aero)

env_for_ctrlr = QuadrotorDiffTrackingEnv(num_eval_envs, 
                              initial_states=x0, 
                              trajectory=eval_trajectory,
                              quad_params=dict(quad_params), 
                              max_time=5, 
                              world=world,
                              control_mode=control_mode, 
                              device=device,
                              render_mode="None",
                              reward_fn=reward_fn,
                              reset_options=eval_reset_options,
                              action_history_length=action_history_length,
                              pos_history_length=pos_history_length,
                              trace_dynamics=True,
                              traj_lookahead_length=lookahead_length,
                              aero=aero)

policy_obs = env_for_policy.reset()
ctrlr_obs = env_for_ctrlr.reset()

terminated = [False for i in range(num_eval_envs)]

params = BatchedMultirotorParams([quad_params] * num_eval_envs, num_eval_envs, device)
controller = BatchedSE3Control(params, num_eval_envs, device)

policy_states = []
policy_actions = np.zeros((500, num_eval_envs, 4))
ctrlr_states = []
ctrlr_actions = np.zeros((500, num_eval_envs, 4))
reference_states = []

# Step and render the environment, comparing the RL agent to the SE3 controller.
t = 0
while t < 500:
    env_for_policy.render()
    ctrlr_state = env_for_ctrlr.vehicle_states
    reference_states.append(eval_trajectory.update(t*0.01)['x'])
    # print(eval_trajectory.update(t*0.01)['x_dot'])
    control_dict = controller.update(0, ctrlr_state, eval_trajectory.update(t*0.01))

    # rescale controller actions to [-1, 1]
    ctrl_norm_thrust = (control_dict["cmd_thrust"].numpy() - 4 * env_for_ctrlr.min_thrust) / (4 * env_for_ctrlr.max_thrust - 4 * env_for_ctrlr.min_thrust)
    ctrl_norm_thrust = ctrl_norm_thrust * 2 - 1
    eulers = roma.unitquat_to_euler('xyz', control_dict["cmd_q"]).numpy()
    eulers_norm = 2 * (eulers + np.pi) / (2 * np.pi) - 1
    ctrlr_action = np.hstack([ctrl_norm_thrust, eulers_norm])
    ctrlr_obs, ctrlr_rwd, ctrlr_done, _ = env_for_ctrlr.step(ctrlr_action)
    ctrlr_states.append(env_for_ctrlr.vehicle_states['x'])
    ctrlr_actions[t] = np.hstack([control_dict["cmd_thrust"].numpy(), eulers])

    # Now do the policy
    policy_action = model.predict(policy_obs, deterministic=True)[0]
    policy_control_dict = env_for_policy.rescale_action(policy_action)
    policy_eulers = R.from_quat(policy_control_dict["cmd_q"]).as_euler('xyz')

    policy_obs, policy_rwd, policy_done, _ = env_for_policy.step(policy_action)
    for i in range(num_eval_envs):
        if policy_done[i]:
            terminated[i] = True
    policy_states.append(env_for_policy.vehicle_states['x'])
    policy_actions[t] = np.hstack([policy_control_dict["cmd_thrust"], policy_eulers])
    t += 1
    print(t)

env_for_policy.close()
env_for_ctrlr.close()

policy_states = np.array(policy_states)
ctrlr_states = np.array(ctrlr_states)
reference_states = np.array(reference_states)

# Plot the results
fig, ax = plt.subplots(4, num_eval_envs, figsize=(10, 2))
for i in range(num_eval_envs):
    ax[0][i].plot(policy_actions[:, i, 0], label="policy")
    ax[0][i].plot(ctrlr_actions[:, i, 0], label="ctrlr")
    ax[0][i].set_title(f"Cmd Thrust Env {i}")
    ax[0][i].legend()
    for j in range(3):
        ax[j+1][i].plot(policy_actions[:, i, j+1], label="policy")
        ax[j+1][i].plot(ctrlr_actions[:, i, j+1], label="ctrlr")
        ax[j+1][i].set_title(f"Cmd Euler {j} Env {i}")
        ax[j+1][i].legend()
fig.tight_layout()
state_fig, ax = plt.subplots(3, num_eval_envs, figsize=(10,2))
for i in range(num_eval_envs):
    for j in range(3):
        ax[j][i].plot(policy_states[:, i, j], label="policy")
        ax[j][i].plot(ctrlr_states[:, i, j], label="ctrlr")
        ax[j][i].plot(reference_states[:, i, j], label="reference")
        ax[j][i].set_title(f"Axis {j} Env {i}")
        ax[j][i].legend()
# state_fig.tight_layout()
plt.show()