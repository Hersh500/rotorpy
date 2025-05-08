import os
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import roma
from scipy.spatial.transform import Rotation as R

import rotorpy
#from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.
from rotorpy.vehicles.crazyfliebrushless_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorDiffTrackingEnv
from rotorpy.learning.quadrotor_reward_functions import vec_diff_reward_negative
from rotorpy.learning.learning_utils import *
from rotorpy.trajectories.hover_traj import BatchedHoverTraj
from rotorpy.controllers.quadrotor_control import BatchedSE3Control

import stable_baselines3
from stable_baselines3 import PPO                                   # We'll use PPO for training.
from stable_baselines3.ppo.policies import MlpPolicy                # The policy will be represented by an MLP
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from rotorpy.vehicles.multirotor import BatchedMultirotorParams

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learning", "policies", "PPO",
                         "traj_cmd_ctattMay-07-23-32")

model_file = "hover_3973120_steps"
# Load ppo policy
model = PPO.load(os.path.join(model_dir, model_file))

# Load an evaluation environment, with ONE env set aside for the SE3 Control.
device = torch.device("cpu")

num_envs = 5
init_rotor_speed = 1788.53

reward_fn = lambda obs, action: vec_diff_reward_negative(obs, action, weights={'x': 1, 'v': 0.1, 'yaw': 0.0, 'w': 2e-2, 'u': 5e-3, 'u_mag': 1e-3, 'survive': 3})
trajectory = BatchedHoverTraj(num_uavs=num_envs)

# generate random initial conditions
x0 = {'x': torch.rand(num_envs,3, device=device).double() * 4 - 2,
        'v': torch.rand(num_envs, 3, device=device).double() * 0.1, 
        'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_envs, 1).double(),
        'w': torch.zeros(num_envs, 3, device=device).double(),
        'wind': torch.zeros(num_envs, 3, device=device).double(),
        'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_envs, 1).double()}

reset_options = dict(rotorpy.learning.quadrotor_environments.DEFAULT_RESET_OPTIONS)
reset_options["params"] = "random"
reset_options["initial_states"] = x0
reset_options["pos_bound"] = 0.5
reset_options["vel_bound"] = 0.2
reset_options["trajectory"] = "fixed"
control_mode = "cmd_ctatt"
params = BatchedMultirotorParams([quad_params] * num_envs, num_envs, device)

# quad_params["tau_m"] = 0.05
env_for_policy = QuadrotorDiffTrackingEnv(num_envs, 
                              initial_states=x0, 
                              trajectory=trajectory,
                              quad_params=params, 
                              max_time=5, 
                              control_mode=control_mode, 
                              device=device,
                              render_mode="3D",
                              reward_fn=reward_fn,
                              reset_options=reset_options,
                                          action_history_length=3)

env_for_ctrlr = QuadrotorDiffTrackingEnv(num_envs, 
                              initial_states=x0, 
                              trajectory=trajectory,
                              quad_params=params, 
                              max_time=5, 
                              control_mode=control_mode, 
                              device=device,
                              render_mode="None",
                              reward_fn=reward_fn,
                              reset_options=reset_options,
                                         action_history_length=3)

policy_obs = env_for_policy.reset()
ctrlr_obs = env_for_ctrlr.reset()

terminated = [False for i in range(num_envs)]

controller = BatchedSE3Control(params, num_envs, device)

policy_states = []
policy_actions = np.zeros((500, num_envs, 4))
ctrlr_states = []
ctrlr_actions = np.zeros((500, num_envs, 4))

# Step and render the environment, comparing the RL agent to the SE3 controller.
t = 0
while t < 500:
    env_for_policy.render()
    ctrlr_state = {'x': torch.from_numpy(ctrlr_obs[:, 0:3]).double(), 
                   'v': torch.from_numpy(ctrlr_obs[:, 3:6]).double(), 
                   'q': torch.from_numpy(ctrlr_obs[:, 6:10]).double(), 
                   'w': torch.from_numpy(ctrlr_obs[:, 10:13]).double()}
    control_dict = controller.update(0, ctrlr_state, trajectory.update(0))

    # rescale controller actions to [-1, 1]
    ctrl_norm_thrust = (control_dict["cmd_thrust"].numpy() - 4 * env_for_ctrlr.min_thrust) / (4 * env_for_ctrlr.max_thrust - 4 * env_for_ctrlr.min_thrust)
    ctrl_norm_thrust = ctrl_norm_thrust * 2 - 1
    eulers = roma.unitquat_to_euler('xyz', control_dict["cmd_q"]).numpy()
    eulers_norm = 2 * (eulers + np.pi) / (2 * np.pi) - 1
    ctrlr_action = np.hstack([ctrl_norm_thrust, eulers_norm])
    ctrlr_obs, ctrlr_rwd, ctrlr_done, _ = env_for_ctrlr.step(ctrlr_action)
    ctrlr_states.append(ctrlr_obs)
    ctrlr_actions[t] = np.hstack([control_dict["cmd_thrust"].numpy(), eulers])

    # Now do the policy
    policy_action = model.predict(policy_obs, deterministic=True)[0]
    policy_control_dict = env_for_policy.rescale_action(policy_action)
    # policy_eulers = R.from_quat(policy_control_dict["cmd_q"] * np.sign(policy_control_dict["cmd_q"][:,-1].reshape(-1, 1))).as_euler('xyz')
    policy_eulers = R.from_quat(policy_control_dict["cmd_q"]).as_euler('xyz')

    policy_obs, policy_rwd, policy_done, _ = env_for_policy.step(policy_action)
    for i in range(num_envs):
        if policy_done[i]:
            terminated[i] = True
    policy_states.append(policy_obs)
    policy_actions[t] = np.hstack([policy_control_dict["cmd_thrust"], policy_eulers])
    t += 1
    print(t)

env_for_policy.close()
env_for_ctrlr.close()

policy_states = np.array(policy_states)
ctrlr_states = np.array(ctrlr_states)

# Plot the results
fig, ax = plt.subplots(4, num_envs, figsize=(10, 2))
for i in range(num_envs):
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
state_fig, ax = plt.subplots(3, num_envs, figsize=(10,2))
for i in range(num_envs):
    for j in range(3):
        ax[j][i].plot(policy_states[:, i, j], label="policy")
        ax[j][i].plot(ctrlr_states[:, i, j], label="ctrlr")
        ax[j][i].set_title(f"Axis {j} Env {i}")
        ax[j][i].legend()
# state_fig.tight_layout()
plt.show()