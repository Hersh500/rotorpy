import os
from datetime import datetime
import torch

import rotorpy
from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorDiffTrackingEnv
from rotorpy.learning.quadrotor_reward_functions import vec_diff_reward_negative
from rotorpy.learning.learning_utils import *
from rotorpy.trajectories.hover_traj import BatchedHoverTraj


# First we'll set up some directories for saving the policy and logs.
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learning", "policies")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learning", "logs")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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

num_envs = 1024
init_rotor_speed = 1788.53

reward_weights = {'x': 1.0, 'v': 0.2, 'yaw': 0.0, 'w': 1e-3, 'u': 1e-3, 'u_mag': 0e-4, 'survive': 3}
reward_fn = lambda obs, action: vec_diff_reward_negative(obs, action, reward_weights)

trajectory = BatchedHoverTraj(num_uavs=num_envs)
x0 = {'x': torch.zeros(num_envs,3, device=device).double(),
        'v': torch.zeros(num_envs, 3, device=device).double(),
        'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_envs, 1).double(),
        'w': torch.zeros(num_envs, 3, device=device).double(),
        'wind': torch.zeros(num_envs, 3, device=device).double(),
        'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_envs, 1).double()}

randomizations = dict(crazyflie_randomizations)
randomizations["mass"] = [0.026, 0.034]
# randomizations["Ixx"] = [0.8e-5, 1.5e-5]
# randomizations["Iyy"] = [0.8e-5, 1.5e-5]
# randomizations["Izz"] = [2.0e-5, 3.1e-5]
randomizations["kp_att"] = [1000, 1200]
randomizations["kd_att"] = [40, 60]
randomizations["tau_m"] = [0.05, 0.05]

reset_options = dict(rotorpy.learning.quadrotor_environments.DEFAULT_RESET_OPTIONS)
reset_options["params"] = "random"
reset_options["randomization_ranges"] = randomizations
reset_options["pos_bound"] = 2.0 
reset_options["vel_bound"] = 0.5
reset_options["trajectory"] = "fixed"

control_mode = "cmd_ctatt"
quad_params["motor_noise_std"] = 0

env = QuadrotorDiffTrackingEnv(num_envs, 
                              initial_states=x0, 
                              trajectory=trajectory,
                              quad_params=dict(quad_params), 
                              max_time=7, 
                              control_mode=control_mode, 
                              device=device,
                              render_mode="None",
                              reward_fn=reward_fn,
                              reset_options=reset_options)


# Allows Stable Baselines to report accurate reward and episode lengths
wrapped_env = VecMonitor(env)

# Create eval environment - set up initial states and trajectory for eval. These could be different from the training env.
num_eval_envs = 5
radii = np.ones((num_eval_envs,3))
trajectory = BatchedHoverTraj(num_uavs=num_eval_envs)
x0_eval = {'x': torch.zeros(num_eval_envs,3, device=device).double(),
        'v': torch.zeros(num_eval_envs, 3, device=device).double(),
        'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_eval_envs, 1).double(),
        'w': torch.zeros(num_eval_envs, 3, device=device).double(),
        'wind': torch.zeros(num_eval_envs, 3, device=device).double(),
        'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_eval_envs, 1).double()}


eval_reset_options = dict(reset_options)
eval_reset_options["trajectory"] = "fixed"
eval_reset_options["params"] = "fixed"
eval_reset_options["initial_state"] = "random"
eval_reset_options["pos_bound"] = 2.0
eval_reset_options["vel_bound"] = 0.2

eval_env = QuadrotorDiffTrackingEnv(num_eval_envs, 
                              initial_states=x0_eval, 
                              trajectory=trajectory,
                              quad_params=dict(quad_params), 
                              max_time=7, 
                              control_mode=control_mode, 
                              device=device,
                              render_mode="3D",
                              reward_fn=reward_fn,
                              reset_options=eval_reset_options)

wrapped_eval_env = VecMonitor(eval_env)

start_time = datetime.now()
checkpoint_callback = CheckpointCallback(save_freq=max(50000//num_envs, 1), save_path=f"{models_dir}/PPO/traj_cmd_ctatt{start_time.strftime('%b-%d-%H-%M')}/",
                                         name_prefix='hover')

eval_callback = EvalCallback(wrapped_eval_env, eval_freq=1e6//num_envs, deterministic=True, render=True)
model = PPO(MlpPolicy,
            wrapped_env,
            n_steps=16,
            batch_size=1024,
            verbose=1,
            device=device,
            tensorboard_log=log_dir,
            policy_kwargs=dict(optimizer_kwargs=dict(weight_decay=0.00001)))

num_timesteps = 15e6
model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False,
            tb_log_name="PPO-QuadHoverTrajVec_"+control_mode + " " + start_time.strftime('%b-%d-%H-%M'),
            callback=CallbackList([checkpoint_callback, eval_callback]))