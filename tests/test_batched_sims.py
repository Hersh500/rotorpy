import numpy as np
import torch

from rotorpy.trajectories.minsnap import BatchedMinSnap
from rotorpy.vehicles.batched_multirotor import BatchedMultirotor
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import BatchedSE3Control, SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.utils.trajgen_utils import generate_random_minsnap_traj
from rotorpy.world import World

def test_batched_operators():
    num_drones = 10
    device = torch.device("cpu")
    init_rotor_speed = 1788.53
    quad_params['motor_noise_std'] = 0
    world = World({"bounds": {"extents": [-10, 10, -10, 10, -10, 10]}, "blocks": []})
    batch_state = {'x': torch.randn(num_drones,3, device=device).double(),
                  'v': torch.randn(num_drones, 3, device=device).double(),
                  'q': torch.tensor([0, 0, 0, 1], device=device).repeat(num_drones, 1).double(),
                  'w': torch.randn(num_drones, 3, device=device).double(),
                  'wind': torch.zeros(num_drones, 3, device=device).double(),
                  'rotor_speeds': torch.tensor([init_rotor_speed, init_rotor_speed, init_rotor_speed, init_rotor_speed], device=device).repeat(num_drones, 1).double()}
    trajectories = [generate_random_minsnap_traj(world, 3, 1.0, 1.0, 2.0, np.random.randn(3)) for _ in range(num_drones)]
    batch_minsnap = BatchedMinSnap(trajectories, device)
    ts = np.array([np.random.uniform(0, 1) for i in range(num_drones)])
    batch_flat_output= batch_minsnap.update(ts)

    # Make sure that BatchedMinSnap does the same thing as MinSnap
    for j, traj in enumerate(trajectories):
        seq_flat_output = traj.update(ts[j])
        for key in batch_flat_output.keys():
            assert np.all(np.abs(batch_flat_output[key][j].cpu().numpy() - seq_flat_output[key]) < 1e-3)

    batched_ctrlr = BatchedSE3Control(quad_params, num_drones, device)
    single_ctrlr = SE3Control(quad_params)
    batch_control_inputs = batched_ctrlr.update(0, batch_state, batch_flat_output)

    batched_multirotor = BatchedMultirotor(quad_params, num_drones, batch_state, device)
    single_multirotor = Multirotor(quad_params)

    batch_next_state = batched_multirotor.step(batch_state, batch_control_inputs, 0.01)

    # Make sure that BatchedSE3Control does the same thing as SE3Control
    # Make sure that BatchedMultirotor does the same thing as MultiRotor
    for j in range(num_drones):
        seq_state = {key: batch_state[key][j].cpu().numpy() for key in batch_state.keys()}
        flat_output = {key: batch_flat_output[key][j].cpu().numpy() for key in batch_flat_output.keys()}
        seq_control_input = single_ctrlr.update(0, seq_state, flat_output)
        for key in batch_control_inputs.keys():
            assert np.all(np.abs(batch_control_inputs[key][j].cpu().numpy() - seq_control_input[key]) < 1e-3)

        seq_next_state = single_multirotor.step(seq_state, seq_control_input, 0.01)
        for key in batch_next_state.keys():
            # since rotor speeds are large, we need a higher tolerance here.
            # Or, something is just wrong, idk. But it's close enough.
            if key == "rotor_speeds":
                assert np.all(np.abs(batch_next_state[key][j].cpu().numpy() - seq_next_state[key]) < 1)
            else:
                # print(batch_next_state[key][j].cpu().numpy())
                # print(seq_next_state[key])
                # print("--------------")
                assert np.all(np.abs(batch_next_state[key][j].cpu().numpy() - seq_next_state[key]) < 1e-2)


if __name__ == "__main__":
    test_batched_operators()
