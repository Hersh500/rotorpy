import numpy as np
from numpy.linalg import inv, norm
import scipy.integrate
from scipy.spatial.transform import Rotation
from rotorpy.vehicles.hummingbird_params import quad_params

# imports for Batched Dynamics
import torch
from torchdiffeq import odeint
import roma

import time


def quat_dot_torch(quat, omega):
    """
    Parameters:
        quat, (...,[i,j,k,w])
        omega, angular velocity of body in body axes: (...,3)

    Returns
        duat_dot, (...,[i,j,k,w])

    """
    b = quat.shape[0]
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[...,0], quat[...,1], quat[...,2], quat[...,3])
    G = torch.stack([q3, q2, -q1, -q0,
                     -q2, q3, q0, -q1,
                     q1, -q0, q3, -q2], dim=1).view((b, 3, 4))

    quat_dot = 0.5 * torch.transpose(G, 1,2) @ omega.unsqueeze(-1)
    # Augment to maintain unit quaternion.
    quat_err = torch.sum(quat**2, dim=-1) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot.squeeze(-1) - quat_err.unsqueeze(-1) * quat_err_grad
    return quat_dot


class BatchedMultirotorFloatParams:
    """ 
    A container class for various multirotor params. 
    Parameters:
        multirotor_params_list: list of dictionaries containing the parameters for each drone (see vehicles/crazyflie_params.py for example of such a dictionary).
        num_drones: number of drones in the simulation.
        device: the device to use for the simulation (e.g. torch.device('cuda')).
    """
    def __init__(self, multirotor_params_list, num_drones, device):
        assert len(multirotor_params_list) == num_drones
        self.num_drones = num_drones
        self.device = device
        self.mass = torch.tensor([multirotor_params['mass'] for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device) # kg

        self.num_rotors = multirotor_params_list[0]["num_rotors"]
        for multirotor_params in multirotor_params_list:
            assert multirotor_params["num_rotors"] == self.num_rotors

        self.rotor_pos = [multirotor_params["rotor_pos"] for multirotor_params in multirotor_params_list]

        self.rotor_dir_np       = np.array([qp['rotor_directions'] for qp in multirotor_params_list])

        self.extract_geometry()

        # Rotor parameters
        self.rotor_speed_min = torch.tensor([multirotor_params['rotor_speed_min'] for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device) # rad/s
        self.rotor_speed_max = torch.tensor([multirotor_params['rotor_speed_max'] for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device) # rad/s

        self.k_eta = np.array([multirotor_params['k_eta'] for multirotor_params in multirotor_params_list])  # .unsqueeze(-1).to(device)     # thrust coeff, N/(rad/s)**2
        self.k_m = np.array([multirotor_params['k_m'] for multirotor_params in multirotor_params_list])  # .unsqueeze(-1).to(device)
        self.k_flap = torch.tensor([multirotor_params['k_flap'] for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device)   # Flapping moment coefficient Nm/(m/s)
        self.k_h = torch.tensor([multirotor_params['k_h'] for multirotor_params in multirotor_params_list]).unsqueeze(-1).float().to(device)   # translational lift coeff N/(m/s)**2

        # Motor parameters
        self.tau_m           = torch.tensor([multirotor_params['tau_m'] for multirotor_params in multirotor_params_list], device=device).unsqueeze(-1)
        self.motor_noise     = torch.tensor([multirotor_params['motor_noise_std'] for multirotor_params in multirotor_params_list], device=device).unsqueeze(-1) # noise added to the actual motor speed, rad/s / sqrt(Hz)

        # Additional constants.
        self.inertia = torch.from_numpy(np.array([[[qp["Ixx"], qp["Ixy"], qp["Ixz"]],
                                                   [qp["Ixy"], qp["Iyy"], qp["Iyz"]],
                                                   [qp["Ixz"], qp["Iyz"], qp["Izz"]]] for qp in multirotor_params_list])).float().to(device)
        self.rotor_drag_matrix = torch.tensor([[[qp["k_d"],   0,                 0],
                                                [0,          qp["k_d"],          0],
                                                [0,          0,          qp["k_z"]]] for qp in multirotor_params_list], device=device).float()

        self.drag_matrix = torch.tensor([[[qp["c_Dx"],   0,                 0],
                                          [0,          qp["c_Dy"],          0],
                                          [0,          0,          qp["c_Dz"]]] for qp in multirotor_params_list], device=device).float()

        # keep this the same across drones.
        self.g = 9.81 # m/s^2

        self.inv_inertia = torch.linalg.inv(self.inertia).float()
        self.weight = torch.zeros(num_drones, 3, device=device).float()
        self.weight[:,-1] = -self.mass.squeeze(-1) * self.g

        # Control allocation
        k = self.k_m/self.k_eta  # Ratio of torque to thrust coefficient.

        # Below is an automated generation of the control allocator matrix. It assumes that all thrust vectors are aligned
        # with the z axis.
        self.f_to_TM = torch.stack([torch.from_numpy(np.vstack((np.ones((1,self.num_rotors)),
                                                                np.hstack([np.cross(self.rotor_pos[i][key],
                                                                                    np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos[i]]),
                                                                (k[i] * self.rotor_dir_np[i]).reshape(1,-1)))).to(device) for i in range(num_drones)]).float()
        self.k_eta = torch.from_numpy(self.k_eta).unsqueeze(-1).to(device)
        self.k_m = torch.from_numpy(self.k_m).unsqueeze(-1).to(device)
        self.rotor_dir = torch.from_numpy(self.rotor_dir_np).to(device)
        self.TM_to_f = torch.linalg.inv(self.f_to_TM).float()

        # Low-Level Control Gains
        self.k_w = torch.tensor([multirotor_params.get('k_w', 1) for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device)  # The body rate P gain (for cmd_ctbr)
        self.k_v = torch.tensor([multirotor_params.get('k_v', 10) for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device)  # The body rate P gain (for cmd_ctbr)
        self.kp_att = torch.tensor([multirotor_params.get('kp_att', 3000.0) for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device)
        self.kd_att = torch.tensor([multirotor_params.get('kd_att', 360.0) for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device)

    # Update methods that require some additional computations.
    # These are useful for dynamically updating the parameters of this object, for example for domain randomization.
    def update_mass(self, idx, mass):
        """
        Update the mass and weight of the drone at index idx.
        """
        self.mass[idx] = mass
        self.weight[idx,-1] = -mass * self.g

    def update_thrust_and_rotor_params(self, idx, k_eta = None, k_m = None, rotor_pos = None):
        """
        Update k_eta, k_m, and rotor positions of the drone at index idx.
        Some parameters are optional, and if not provided, the current value is kept.
        """
        if k_eta is not None:
            self.k_eta[idx] = k_eta
        if k_m is not None:
            self.k_m[idx] = k_m
        k_idx = self.k_m[idx]/self.k_eta[idx]
        if rotor_pos is not None:
            assert 'r1' in rotor_pos.keys() and 'r2' in rotor_pos.keys() and 'r3' in rotor_pos.keys() and 'r4' in rotor_pos.keys()
            self.rotor_pos[idx] = dict(rotor_pos[k_idx])
            rotor_geometry = np.array([]).reshape(0, 3)
            for rotor in rotor_pos:
                r = rotor_pos[rotor]
                rotor_geometry = np.vstack([rotor_geometry, r])
            self.rotor_geometry[idx] = torch.from_numpy(np.array(rotor_geometry)).float().to(self.device)
            self.rotor_geometry_hat_maps[idx] = BatchedMultirotorFloat.hat_map(torch.from_numpy(rotor_geometry.squeeze())).float().to(self.device)

        self.f_to_TM[idx] = torch.from_numpy(np.vstack((np.ones((1,self.num_rotors)),
                                                                np.hstack([np.cross(self.rotor_pos[idx][key],
                                                                                    np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos[idx]]),
                                                                (k_idx.cpu() * self.rotor_dir_np[idx]).reshape(1,-1)))).to(self.device)
        self.TM_to_f[idx] = torch.linalg.inv(self.f_to_TM[idx])

    def update_inertia(self, idx, Ixx=None, Iyy=None, Izz=None):
        """
        Update the inertia of the drone at index idx. Parameters with value None are not updated.
        """
        self.inertia[idx][0,0] = Ixx
        self.inertia[idx][1,1] = Iyy
        self.inertia[idx][2,2] = Izz
        self.inv_inertia[idx] = torch.linalg.inv(self.inertia[idx])

    def update_drag(self, idx, c_Dx=None, c_Dy=None, c_Dz=None, k_d=None, k_z=None):
        """
        Update various drag parameters of the drone at index idx. Parameters with value None are not updated.
        """
        if c_Dx is not None:
            self.drag_matrix[idx][0,0] = c_Dx
        if c_Dy is not None:
            self.drag_matrix[idx][1,1] = c_Dy
        if c_Dz is not None:
            self.drag_matrix[idx][2,2] = c_Dz
        if k_d is not None:
            self.rotor_drag_matrix[idx][0,0] = k_d
            self.rotor_drag_matrix[idx][1,1] = k_d
        if k_z is not None:
            self.rotor_drag_matrix[idx][2,2] = k_z

    def extract_geometry(self):
        """
        Extracts the geometry in self.rotors for efficient use later on in the computation of
        wrenches acting on the rigid body.
        The rotor_geometry is a tensor of shape (num_drones, n,3), where n is the number of rotors.
        Each row corresponds to the position vector of the rotor relative to the CoM.
        """

        geoms = []
        geom_hat_maps = []
        for i in range(self.num_drones):
            rotor_geometry = np.array([]).reshape(0, 3)
            for rotor in self.rotor_pos[i]:
                r = self.rotor_pos[i][rotor]
                rotor_geometry = np.vstack([rotor_geometry, r])
            geoms.append(rotor_geometry)
            geom_hat_maps.append(BatchedMultirotorFloat.hat_map(torch.from_numpy(rotor_geometry.squeeze())).numpy())
        self.rotor_geometry = torch.from_numpy(np.array(geoms)).float().to(self.device)
        self.rotor_geometry_hat_maps = torch.from_numpy(np.array(geom_hat_maps)).float().to(self.device)


class BatchedMultirotorFloat(object):
    """
    Batched Multirotor forward dynamics model, implemented in PyTorch.
    This class follows the same API as the original Multirotor class, but is designed to work with batches of drones.
    Generally, quantities that would be numpy arrays in `Multirotor` are instead torch tensors in this class.

    states: [position, velocity, attitude, body rates, wind, rotor speeds]

    Parameters:
        batched_params: BatchedMultirotorParams object, containing relevant physical parameters for the multirotor.
        num_drones: the number of drones in the batch.
        initial_states: the initial state of the vehicle. Contains the same keys as "initial_states" of `Multirotor`, but each 
                                value is a pytorch tensor with a prepended batch dimension. e.g. initial_states['x'].shape = (num_drones, 3).
                                To maintain fidelity during the simulation, the expected datatype for the tensors is float.
        control_abstraction: the appropriate control abstraction that is used by the controller, options are...
                                'cmd_motor_speeds': the controller directly commands motor speeds.
                                'cmd_motor_thrusts': the controller commands forces for each rotor.
                                'cmd_ctbr': the controller commands a collective thrsut and body rates.
                                'cmd_ctbm': the controller commands a collective thrust and moments on the x/y/z body axes
                                'cmd_ctatt': the controller commands a collective thrust and attitude (as a quaternion).
                                'cmd_vel': the controller commands a velocity vector in the world frame.
                                'cmd_acc': the controller commands a mass normalized thrust vector (acceleration) in the world frame.
        aero: boolean, determines whether or not aerodynamic drag forces are computed.
        integrator: str, "dopri5" or "rk4", which are adaptive or fixed step size integrators. "rk4" will be faster, but potentially less accurate.
    """

    def __init__(self, batched_params,
                 num_drones,
                 initial_states,
                 device,
                 control_abstraction='cmd_motor_speeds',
                 aero=True,
                 integrator='dopri5'
                 ):
        """
        Initialize quadrotor physical parameters.
        """
        assert initial_states['x'].device == device, "Initial states must already be on the specified device."
        assert initial_states['x'].shape[0] == num_drones
        assert batched_params.device == device
        self.num_drones = num_drones
        self.device = device
        self.params = batched_params

        # Set the initial state
        self.initial_states = initial_states
        self.control_abstraction = control_abstraction

        self.aero = aero

        # assert integrator == 'dopri5' or integrator == "rk4"
        self.integrator = integrator

    def statedot(self, state, control, t_step, idxs):
        """
        Integrate dynamics forward from state given constant cmd_rotor_speeds for time t_step.
        Returns:
            vdot: torch.tensor of shape (num_drones, 3), with zeros for drones not in idxs
            wdot: torch.tensor of shape (num_drones, 3), with zeros for drones not in idxs
        """

        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control, idxs)

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = torch.clip(cmd_rotor_speeds, self.params.rotor_speed_min[idxs],
                                      self.params.rotor_speed_max[idxs])

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds, idxs)

        s = BatchedMultirotorFloat._pack_state(state, self.num_drones, self.device)

        s_dot = s_dot_fn(0, s[idxs])
        v_dot = torch.zeros_like(state["v"])
        w_dot = torch.zeros_like(state["w"])
        v_dot[idxs] = s_dot[..., 3:6].float()
        w_dot[idxs] = s_dot[..., 10:13].float()

        state_dot = {'vdot': v_dot, 'wdot': w_dot}
        return state_dot

    def step(self, state, control, t_step, idxs=None):
        """
        Integrate dynamics forward from state given constant control for time t_step.
        Params:
            - state: dictionary containing keys ['x', 'v', 'q', 'w', 'wind', 'rotor_speeds'], and values which are pytorch tensors with dtype floatand which have a batch dimension
            - control: dictionary with keys depending on the chosen control mode. Values are torch tensors, again with dtype float and with a batch dimension equal to the number of drones.
            - t_step: float, the duration for which to step the simulation.
            - idxs: integer array of shape (num_running_drones, )
        """
        if idxs is None:
            idxs = [i for i in range(self.num_drones)]
        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control, idxs)

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = torch.clip(cmd_rotor_speeds, self.params.rotor_speed_min[idxs],
                                      self.params.rotor_speed_max[idxs])

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds, idxs)

        s = BatchedMultirotorFloat._pack_state(state, self.num_drones, self.device)

        # Option 1 - RK45 integration
        # sol = scipy.integrate.solve_ivp(s_dot_fn, (0, t_step), s, first_step=t_step)
        with torch.no_grad():
            sol = odeint(s_dot_fn, s[idxs], t=torch.tensor([0.0, t_step], device=self.device), method=self.integrator)
        # s = sol['y'][:,-1]
        s = sol[-1, :]
        # Option 2 - Euler integration
        # s = s + s_dot_fn(0, s) * t_step  # first argument doesn't matter. It's time invariant model

        state = BatchedMultirotorFloat._unpack_state(s, idxs, self.num_drones)

        # Re-normalize unit quaternion.
        state['q'][idxs] = state['q'][idxs] / torch.norm(state['q'][idxs], dim=-1).unsqueeze(-1)

        # Add noise to the motor speed measurement
        state['rotor_speeds'][idxs] += torch.normal(mean=torch.zeros(self.params.num_rotors, device=self.device),
                                                    std=torch.ones(len(idxs), self.params.num_rotors,
                                                                   device=self.device) * torch.abs(
                                                        self.params.motor_noise[idxs]))
        state['rotor_speeds'][idxs] = torch.clip(state['rotor_speeds'][idxs], self.params.rotor_speed_min[idxs],
                                                 self.params.rotor_speed_max[idxs])

        return state

    # Cmd rotor speeds should already have the appropriate drones selected.
    def _s_dot_fn(self, t, s, cmd_rotor_speeds, idxs):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """

        # so this will be zero for some stuff.
        state = BatchedMultirotorFloat._unpack_state(s, idxs, self.num_drones)

        rotor_speeds = state['rotor_speeds'][idxs]
        inertial_velocity = state['v'][idxs]
        wind_velocity = state['wind'][idxs]

        # R = Rotation.from_quat(state['q']).as_matrix()
        R = roma.unitquat_to_rotmat(state['q'][idxs]).float()

        # Rotor speed derivative
        rotor_accel = (1 / self.params.tau_m[idxs]) * (cmd_rotor_speeds - rotor_speeds)

        # Position derivative.
        x_dot = state['v'][idxs]

        # Orientation derivative.
        q_dot = quat_dot_torch(state['q'][idxs], state['w'][idxs])

        # Compute airspeed vector in the body frame
        body_airspeed_vector = R.transpose(1, 2) @ (inertial_velocity - wind_velocity).unsqueeze(-1).float()
        body_airspeed_vector = body_airspeed_vector.squeeze(-1)

        # Compute total wrench in the body frame based on the current rotor speeds and their location w.r.t. CoM
        (FtotB, MtotB) = self.compute_body_wrench(state['w'][idxs], rotor_speeds, body_airspeed_vector, idxs)

        # Rotate the force from the body frame to the inertial frame
        Ftot = R @ FtotB.unsqueeze(-1)

        # Velocity derivative.
        v_dot = (self.params.weight[idxs] + Ftot.squeeze(-1)) / self.params.mass[idxs]

        # Angular velocity derivative.
        w = state['w'][idxs].float()
        w_hat = BatchedMultirotorFloat.hat_map(w).permute(2, 0, 1)
        w_dot = self.params.inv_inertia[idxs] @ (
                    MtotB - (w_hat.float() @ (self.params.inertia[idxs] @ w.unsqueeze(-1))).squeeze(-1)).unsqueeze(-1)

        # NOTE: the wind dynamics are currently handled in the wind_profile object.
        # The line below doesn't do anything, as the wind state is assigned elsewhere.
        wind_dot = torch.zeros((len(idxs), 3), device=self.device)

        # Pack into vector of derivatives.
        s_dot = torch.zeros((len(idxs), 16 + self.params.num_rotors,), device=self.device)
        s_dot[:, 0:3] = x_dot
        s_dot[:, 3:6] = v_dot
        s_dot[:, 6:10] = q_dot
        s_dot[:, 10:13] = w_dot.squeeze(-1)
        s_dot[:, 13:16] = wind_dot
        s_dot[:, 16:] = rotor_accel

        return s_dot

    def compute_body_wrench(self, body_rates, rotor_speeds, body_airspeed_vector, idxs):
        """
        Computes the wrench acting on the rigid body based on the rotor speeds for thrust and airspeed
        for aerodynamic forces.
        The airspeed is represented in the body frame.
        The net force Ftot is represented in the body frame.
        The net moment Mtot is represented in the body frame.
        """
        assert (body_rates.shape[0] == rotor_speeds.shape[0])
        assert (body_rates.shape[0] == body_airspeed_vector.shape[0])

        num_drones = body_rates.shape[0]
        # Get the local airspeeds for each rotor
        local_airspeeds = body_airspeed_vector.unsqueeze(-1) + (
            BatchedMultirotorFloat.hat_map(body_rates).permute(2, 0, 1)) @ (self.params.rotor_geometry[idxs].transpose(1, 2))

        # Compute the thrust of each rotor, assuming that the rotors all point in the body z direction!
        T = torch.zeros(num_drones, 3, 4, device=self.device)
        T[..., -1, :] = self.params.k_eta[idxs] * rotor_speeds ** 2

        # Add in aero wrenches (if applicable)
        if self.aero:
            # Parasitic drag force acting at the CoM
            tmp = self.params.drag_matrix[idxs] @ (body_airspeed_vector).unsqueeze(-1)
            D = -BatchedMultirotorFloat._norm(body_airspeed_vector).unsqueeze(-1) * tmp.squeeze()
            # Rotor drag (aka H force) acting at each propeller hub.
            tmp = self.params.rotor_drag_matrix[idxs] @ local_airspeeds.float()
            H = -rotor_speeds.unsqueeze(1) * tmp
            # Pitching flapping moment acting at each propeller hub.
            M_flap = BatchedMultirotorFloat.hat_map(local_airspeeds.transpose(1, 2).reshape(num_drones * 4, 3))
            M_flap = M_flap.permute(2, 0, 1).reshape(num_drones, 4, 3, 3).float()
            M_flap = M_flap @ torch.tensor([0, 0, 1.0], device=self.device).float()
            M_flap = (-self.params.k_flap[idxs] * rotor_speeds).unsqueeze(1) * M_flap.transpose(-1, -2)

            lift = torch.zeros(num_drones, 3, 1, device=self.device).float()
            lift[:, 2, :] = self.params.k_h[idxs]
            lift = torch.bmm(lift, (local_airspeeds[:, 0, :] ** 2 + local_airspeeds[:, 1, :] ** 2).unsqueeze(1))
            T += lift
            D = torch.zeros(num_drones, 3, device=self.device).float()
            H = torch.zeros((num_drones, 3, self.params.num_rotors), device=self.device).float()
            M_flap = torch.zeros((num_drones, 3, self.params.num_rotors), device=self.device).float()

        # Compute the moments due to the rotor thrusts, rotor drag (if applicable), and rotor drag torques
        M_force = -torch.einsum('bijk, bik->bj', self.params.rotor_geometry_hat_maps[idxs], T + H)
        M_yaw = torch.zeros(num_drones, 3, 4, device=self.device)
        M_yaw[..., -1, :] = self.params.rotor_dir[idxs] * self.params.k_m[idxs] * rotor_speeds ** 2

        # Sum all elements to compute the total body wrench
        FtotB = torch.sum(T + H, dim=2) + D
        MtotB = M_force + torch.sum(M_yaw + M_flap, dim=2)

        return (FtotB, MtotB)

    # FIXME(hersh500): since so much of this code is shared with the SE3 Controller, it should really be
    # cleaned up and split into different functions that can be shared across both objects.
    def get_cmd_motor_speeds(self, state, control, idxs):
        """
        Computes the commanded motor speeds depending on the control abstraction.
        For higher level control abstractions, we have low-level controllers that will produce motor speeds based on the higher level commmand.
        Currently, we are naively casting everything to float, but ideally the controller should just output floats.
        """

        if self.control_abstraction == 'cmd_motor_speeds':
            # The controller directly controls motor speeds, so command that.
            return control['cmd_motor_speeds'][idxs].float()
        elif self.control_abstraction == "cmd_motor_thrusts":
            cmd_motor_speeds = control["cmd_motor_thrusts"][idxs].float() / self.params.k_eta[idxs]
            return torch.sign(cmd_motor_speeds) * torch.sqrt(torch.abs(cmd_motor_speeds))
        elif self.control_abstraction == "cmd_ctbm":
            cmd_thrust = control['cmd_thrust'][idxs].float()
            cmd_moment = control['cmd_moment'][idxs].float()
        elif self.control_abstraction == "cmd_ctbr":
            cmd_thrust = control['cmd_thrust'][idxs].float()

            # First compute the error between the desired body rates and the actual body rates given by state.
            w_err = state['w'][idxs].float() - control['cmd_w'][idxs].float()

            # Computed commanded moment based on the attitude error and body rate error
            wdot_cmd = -self.params.k_w[idxs] * w_err
            cmd_moment = self.params.inertia[idxs] @ wdot_cmd.unsqueeze(-1)
        elif self.control_abstraction == "cmd_vel":
            # The controller commands a velocity vector.
            # Get the error in the current velocity.
            v_err = state['v'][idxs].float() - control['cmd_v'][idxs].float()

            # Get desired acceleration based on P control of velocity error.
            a_cmd = -self.params.k_v[idxs] * v_err

            # Get desired force from this acceleration.
            F_des = self.params.mass[idxs] * (a_cmd + np.array([0, 0, self.params.g]))

            R = roma.unitquat_to_rotmat(state['q'][idxs]).float()
            b3 = R @ torch.tensor([0.0, 0.0, 1.0], device=self.device).float()
            cmd_thrust = torch.sum(F_des * b3, dim=-1).float().unsqueeze(-1)

            # Follow rest of SE3 controller to compute cmd moment.
            # Desired orientation to obtain force vector.
            b3_des = F_des / torch.norm(F_des, dim=-1, keepdim=True)
            c1_des = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).float()
            b2_des = torch.cross(b3_des, c1_des, dim=-1) / torch.norm(torch.cross(b3_des, c1_des, dim=-1), dim=-1,
                                                                      keepdim=True)
            b1_des = torch.cross(b2_des, b3_des, dim=-1)
            R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)

            # Orientation error.
            S_err = 0.5 * (R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des)
            att_err = torch.stack([-S_err[:, 1, 2], S_err[:, 0, 2], -S_err[:, 0, 1]], dim=-1)

            # Angular control; vector units of N*m.
            Iw = self.params.inertia[idxs] @ state['w'][idxs].unsqueeze(-1).float()
            tmp = -self.params.kp_att[idxs] * att_err - self.params.kd_att[idxs] * state['w']
            cmd_moment = (self.params.inertia[idxs] @ tmp.unsqueeze(-1)).squeeze(-1) + torch.cross(state['w'][idxs],
                                                                                                   Iw.squeeze(-1),
                                                                                                   dim=-1)
        elif self.control_abstraction == "cmd_ctatt":
            cmd_thrust = control["cmd_thrust"][idxs].float()
            R = roma.unitquat_to_rotmat(state['q'][idxs]).float()
            R_des = roma.unitquat_to_rotmat(control["cmd_q"][idxs]).float()
            S_err = 0.5 * (R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des)
            att_err = torch.stack([-S_err[:, 1, 2], S_err[:, 0, 2], -S_err[:, 0, 1]], dim=-1)
            Iw = self.params.inertia[idxs] @ state['w'][idxs].unsqueeze(-1).float()
            tmp = -self.params.kp_att[idxs] * att_err - self.params.kd_att[idxs] * state['w'][idxs]
            cmd_moment = (self.params.inertia[idxs] @ tmp.unsqueeze(-1)).squeeze(-1) + torch.cross(state['w'][idxs],
                                                                                                   Iw.squeeze(-1),
                                                                                                   dim=-1)
        elif self.control_abstraction == "cmd_acc":
            F_des = control['cmd_acc'][idxs].float() * self.params.mass[idxs]
            R = roma.unitquat_to_rotmat(state['q'][idxs]).float()
            b3 = R @ torch.tensor([0.0, 0.0, 1.0], device=self.device).float()
            cmd_thrust = torch.sum(F_des * b3, dim=-1).float().unsqueeze(-1)

            # Follow rest of SE3 controller to compute cmd moment.
            # Desired orientation to obtain force vector.
            b3_des = F_des / torch.norm(F_des, dim=-1, keepdim=True)
            c1_des = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).float()
            b2_des = torch.cross(b3_des, c1_des, dim=-1) / torch.norm(torch.cross(b3_des, c1_des, dim=-1), dim=-1,
                                                                      keepdim=True)
            b1_des = torch.cross(b2_des, b3_des, dim=-1)
            R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)

            # Orientation error.
            S_err = 0.5 * (R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des)
            att_err = torch.stack([-S_err[:, 1, 2], S_err[:, 0, 2], -S_err[:, 0, 1]], dim=-1)

            # Angular control; vector units of N*m.
            Iw = self.params.inertia[idxs] @ state['w'][idxs].unsqueeze(-1).float()
            tmp = -self.params.kp_att[idxs] * att_err - self.params.kd_att[idxs] * state['w']
            cmd_moment = (self.params.inertia[idxs] @ tmp.unsqueeze(-1)).squeeze(-1) + torch.cross(state['w'][idxs],
                                                                                                   Iw.squeeze(-1),
                                                                                                   dim=-1)
        else:
            raise ValueError(
                "Invalid control abstraction selected. Options are: cmd_motor_speeds, cmd_motor_thrusts, cmd_ctbm, cmd_ctbr, cmd_ctatt, cmd_vel, cmd_acc")

        TM = torch.cat([cmd_thrust, cmd_moment.squeeze(-1)], dim=-1)
        cmd_rotor_thrusts = (self.params.TM_to_f[idxs] @ TM.unsqueeze(1).transpose(-1, -2)).squeeze(-1)
        cmd_motor_speeds = cmd_rotor_thrusts / self.params.k_eta[idxs]
        cmd_motor_speeds = torch.sign(cmd_motor_speeds) * torch.sqrt(torch.abs(cmd_motor_speeds))
        return cmd_motor_speeds

    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array([2 * (q[0] * q[2] + q[1] * q[3]),
                         2 * (q[1] * q[2] - q[0] * q[3]),
                         1 - 2 * (q[0] ** 2 + q[1] ** 2)])

    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        In the vectorized implementation, we assume that s is in the shape (N arrays, 3)
        """
        device = s.device
        if len(s.shape) > 1:  # Vectorized implementation
            s = s.unsqueeze(-1)
            hat = torch.cat([torch.zeros(s.shape[0], 1, device=device), -s[:, 2], s[:, 1],
                             s[:, 2], torch.zeros(s.shape[0], 1, device=device), -s[:, 0],
                             -s[:, 1], s[:, 0], torch.zeros(s.shape[0], 1, device=device)], dim=0).view(3, 3, s.shape[
                0]).float()
            return hat
        else:
            return torch.tensor([[0, -s[2], s[1]],
                                 [s[2], 0, -s[0]],
                                 [-s[1], s[0], 0]], device=device)

    @classmethod
    def _pack_state(cls, state, num_drones, device):
        """
        Convert a state dict to Quadrotor's private internal vector representation.
        """
        s = torch.zeros(num_drones, 20,
                        device=device).float()  # FIXME: this shouldn't be hardcoded. Should vary with the number of rotors.
        s[..., 0:3] = state['x']  # inertial position
        s[..., 3:6] = state['v']  # inertial velocity
        s[..., 6:10] = state['q']  # orientation
        s[..., 10:13] = state['w']  # body rates
        s[..., 13:16] = state['wind']  # wind vector
        s[..., 16:] = state['rotor_speeds']  # rotor speeds

        return s

    @classmethod
    def _norm(cls, v):
        """
        Given a vector v in R^3, return the 2 norm (length) of the vector
        """
        # norm = (v[...,0]**2 + v[...,1]**2 + v[...,2]**2)**0.5
        norm = torch.linalg.norm(v, dim=-1)
        return norm

    @classmethod
    def _unpack_state(cls, s, idxs, num_drones):
        """
        Convert Quadrotor's private internal vector representation to a state dict.
        x = inertial position
        v = inertial velocity
        q = orientation
        w = body rates
        wind = wind vector
        rotor_speeds = rotor speeds
        """
        device = s.device
        state = {'x': torch.full((num_drones, 3), float("nan"), device=device).float(),
                 'v': torch.full((num_drones, 3), float("nan"), device=device).float(),
                 'q': torch.full((num_drones, 4), float("nan"), device=device).float(),
                 'w': torch.full((num_drones, 3), float("nan"), device=device).float(),
                 'wind': torch.full((num_drones, 3), float("nan"), device=device).float(),
                 'rotor_speeds': torch.full((num_drones, 4), float("nan"), device=device).float()}
        state['q'][..., -1] = 1  # make sure we're returning a valid quaternion
        state['x'][idxs] = s[:, 0:3]
        state['v'][idxs] = s[:, 3:6]
        state['q'][idxs] = s[:, 6:10]
        state['w'][idxs] = s[:, 10:13]
        state['wind'][idxs] = s[:, 13:16]
        state['rotor_speeds'][idxs] = s[:, 16:]
        return state