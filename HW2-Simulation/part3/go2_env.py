import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                # for this locomotion policy there are usually no more than 30 collision pairs
                # set a low value can save memory
                max_collision_pairs=30,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # add terrain
        self.build_terrain()

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.lin_vel_x_sums = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

    def build_terrain(self):
        ...

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        # TODO: update all relevant buffers (base position, orientation, velocities, DOF states, projected gravity)
        # Hint: you can refer to self.robot.get_pos(), get_quat(), get_vel(), get_ang(), etc.
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        inv_init_q = self.inv_base_init_quat.unsqueeze(0).expand_as(self.base_quat)
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(inv_init_q, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        # reset conditions to avoid weird states
        self.reset_buf |= torch.abs(self.base_pos[:, 2]) > 1.0 # Reset if too high
        self.reset_buf |= torch.abs(self.base_pos[:, 2]) < 0.2 # Reset if too low
        self.reset_buf |= torch.any(torch.abs(self.base_lin_vel) > 5.0, dim=1) # Reset if moving too fast
        self.reset_buf |= torch.any(torch.abs(self.base_ang_vel) > 5.0, dim=1) # Reset if spinning too fast
        self.reset_buf |= torch.any(torch.abs(self.dof_vel) > 15.0, dim=1) # Reset if joints move too fast

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        self.lin_vel_x_sums += self.base_lin_vel[:, 0]

        self.rew_buf = torch.nan_to_num(self.rew_buf)
        self.rew_buf = torch.clamp(self.rew_buf, min=-10.0, max=10.0)

        # compute observations
        # TODO: construct observation buffer self.obs_buf by concatenating key features
        # e.g., base angular velocity, projected gravity, commands, DOF positions/velocities, actions

        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
        ], dim=-1)
        self.obs_buf.nan_to_num_().clamp_(min=-100.0, max=100.0)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def set_command(self, lin_vel_x, lin_vel_y, ang_vel):
        self.commands[:, 0] = lin_vel_x
        self.commands[:, 1] = lin_vel_y
        self.commands[:, 2] = ang_vel

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        """
        Reset the selected environments.

        TODO: Implement the environment reset procedure.
        Steps typically include:
            1. Reset DOF positions and velocities to default.
            2. Reset robot base position and orientation.
            3. Clear velocity and buffer variables.
            4. Reset episode length and termination flags.
            5. Reinitialize per-episode statistics and resample commands.
        """
        if len(envs_idx) == 0:
            return

        self.dof_pos[envs_idx] = self.default_dof_pos.unsqueeze(0).expand(len(envs_idx), -1)
        self.dof_vel[envs_idx] = 0.0
        self.base_pos[envs_idx] = self.base_init_pos.unsqueeze(0).expand(len(envs_idx), -1)
        self.base_quat[envs_idx] = self.base_init_quat.unsqueeze(0).expand(len(envs_idx), -1)
        self.base_lin_vel[envs_idx] = 0.0
        self.base_ang_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.default_dof_pos.unsqueeze(0).expand(len(envs_idx), -1),
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx)
        self.robot.set_pos(self.base_init_pos.unsqueeze(0).expand(len(envs_idx), -1), zero_velocity=True, envs_idx=envs_idx)
        self.robot.set_quat(self.base_init_quat.unsqueeze(0).expand(len(envs_idx), -1), zero_velocity=True, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)
        
        self.extras["episode"] = {}
        for k in self.episode_sums.keys():
            self.extras["episode"][k + "_reward"] = (
                torch.mean(self.episode_sums[k][envs_idx]).item()
            )
            self.episode_sums[k][envs_idx] = 0.0
        self.extras["episode"]["mean_lin_vel_x"] = torch.mean(self.lin_vel_x_sums[envs_idx] / torch.clamp(self.episode_length_buf[envs_idx].float(), min=1.0)).item()

        self.lin_vel_x_sums[envs_idx] = 0.0
        self.reset_buf[envs_idx] = True
        self.episode_length_buf[envs_idx] = 0
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        """
        Tracking of linear velocity commands (x, y axes)
        Formula:
            r = exp(-||v_cmd_xy - v_base_xy||² / tracking_sigma)
        """
        # TODO: implement this formula using torch operations
        error = torch.sum((self.commands[:, :2] - self.base_lin_vel[:, :2]) ** 2, dim=1)
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        """
        Tracking of angular velocity commands (yaw)
        Formula:
            r = exp(-(ω_cmd - ω_base)² / tracking_sigma)
        """
        error = (self.commands[:, 2] - self.base_ang_vel[:, 2]) ** 2
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        """
        Penalize z-axis linear velocity.
        Formula:
            r = (v_z)²
        """
        return self.base_lin_vel[:, 2] ** 2

    def _reward_action_rate(self):
        """
        Penalize changes in actions.
        Formula:
            r = Σ (a_t - a_{t-1})²
        """
        return torch.sum((self.last_actions - self.actions) ** 2, dim=1)

    def _reward_similar_to_default(self):
        """
        Penalize deviation from default joint positions.
        Formula:
            r = Σ |q - q_default|
        """
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        """
        Penalize deviation of base height from target.
        Formula:
            r = (z - h_target)²
        """
        return (self.base_pos[:, 2] - self.reward_cfg["base_height_target"]) ** 2

    def _reward_alive(self):
        return torch.ones(self.num_envs, device=self.device, dtype=gs.tc_float)
