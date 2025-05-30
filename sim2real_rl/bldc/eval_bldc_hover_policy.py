import os
from datetime import datetime
import torch
import roma
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import stable_baselines3
from stable_baselines3 import PPO                                   # We'll use PPO for training.
from stable_baselines3.ppo.policies import MlpPolicy                # The policy will be represented by an MLP
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

import rotorpy
from rotorpy.vehicles.crazyfliebrushless_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorDiffTrackingEnv
from rotorpy.learning.quadrotor_reward_functions import vec_diff_reward_negative
from rotorpy.learning.learning_utils import *
from rotorpy.trajectories.hover_traj import BatchedHoverTraj
from rotorpy.trajectories.circular_traj import BatchedThreeDCircularTraj
from rotorpy.controllers.quadrotor_control import BatchedSE3Control
from rotorpy.world import World


model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "learning", "policies", "PPO",
                         "hover_bldc_cmd_ctattMay-28-17-50")

model_file = "hover_bldc_4079616_steps"

# Load ppo policy
model = PPO.load(os.path.join(model_dir, model_file))
device = torch.device("cpu")
num_eval_envs = 5
control_mode = "cmd_ctatt"
action_history_length = 3
pos_history_length = 3
lookahead_length = 0

reset_options = dict(rotorpy.learning.quadrotor_environments.DEFAULT_RESET_OPTIONS)
eval_reset_options = dict(reset_options)
eval_reset_options["traj_randomization_fn"] = None
eval_reset_options["params"] = "fixed"
eval_reset_options["initial_state"] = "fixed"


eval_trajectory = BatchedHoverTraj(num_uavs=num_eval_envs)

reward_weights = {'x': 1.0, 
                  'v': 0.4, 
                  'yaw': 0.5, 
                  'w': 0e-3, 
                  'u': np.array([1e-3, 1e-3, 1e-3, 1e-3]), 
                  'u_mag': np.array([0e-4, 0e-4, 0e-4, 0e-4]), 
                  'survive': 5}

reward_fn = lambda obs, action, **kwargs: vec_diff_reward_negative(obs, action, reward_weights, **kwargs)
init_rotor_speed = 1788.53

# generate random initial conditions
x0 = {'x': torch.rand(num_eval_envs,3, device=device).double() * 4 - 2,
        'v': torch.rand(num_eval_envs, 3, device=device).double() * 0.1, 
        'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_eval_envs, 1).double(),
        'w': torch.zeros(num_eval_envs, 3, device=device).double(),
        'wind': torch.zeros(num_eval_envs, 3, device=device).double(),
        'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_eval_envs, 1).double()}

wbound = 5
world = World.empty((-wbound, wbound, -wbound,
                     wbound, -wbound, wbound))

params = BatchedMultirotorParams([quad_params] * num_eval_envs, num_eval_envs, device)

env_for_policy = QuadrotorDiffTrackingEnv(num_eval_envs, 
                              initial_states=x0, 
                              trajectory=eval_trajectory,
                              quad_params=params, 
                              max_time=10, 
                              world=world,
                              control_mode=control_mode, 
                              device=device,
                              render_mode="3D",
                              reward_fn=reward_fn,
                              reset_options=eval_reset_options,
                              action_history_length=action_history_length,
                              pos_history_length=pos_history_length,
                              traj_lookahead_length=lookahead_length,
                              trace_dynamics=False)

env_for_ctrlr = QuadrotorDiffTrackingEnv(num_eval_envs, 
                              initial_states=x0, 
                              trajectory=eval_trajectory,
                              quad_params=params, 
                              max_time=10, 
                              world=world,
                              control_mode=control_mode, 
                              device=device,
                              render_mode="None",
                              reward_fn=reward_fn,
                              reset_options=eval_reset_options,
                              action_history_length=action_history_length,
                              pos_history_length=pos_history_length,
                              traj_lookahead_length=lookahead_length,
                              trace_dynamics=False)

policy_obs = env_for_policy.reset()
ctrlr_obs = env_for_ctrlr.reset()

terminated = [False for i in range(num_eval_envs)]

controller = BatchedSE3Control(params, num_eval_envs, device, kp_att=torch.tensor([quad_params["kp_att"]], device=device).repeat(num_eval_envs, 1).double(), 
                               kd_att=torch.tensor([quad_params["kd_att"]], device=device).repeat(num_eval_envs, 1).double())
num_eval_steps = 1000
policy_states = []
policy_actions = np.zeros((num_eval_steps, num_eval_envs, 4))
ctrlr_states = []
ctrlr_actions = np.zeros((num_eval_steps, num_eval_envs, 4))

reference_states = []

# Step and render the environment, comparing the RL agent to the SE3 controller.
t = 0
while t < num_eval_steps:
    env_for_policy.render()
    reference_states.append(eval_trajectory.update(t*0.01)['x'])
    control_dict = controller.update(t*0.01, env_for_ctrlr.vehicle_states, eval_trajectory.update(t*0.01))

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
plt.show()