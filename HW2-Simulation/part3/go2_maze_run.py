import pickle
import numpy as np
import torch
import genesis as gs
from pathlib import Path
from dataclasses import dataclass
import tyro
from rsl_rl.runners import OnPolicyRunner
from go2_maze import Go2MazeEnv, MAZE

@dataclass
class NavigationCFG:
    exp_name: str = "go2-locomotion"
    ckpt: int = 300
    output_dir: Path = Path("part3")
    maze_scale: float = 1.0
    wall_height: float = 1.5
    camera_height: float = 8.0
    camera_res: tuple[int, int] = (1280, 720)
    max_steps: int = 4000
    waypoint_tol: float = 0.25
    lin_gain: float = 1.0
    ang_gain: float = 2.5
    max_lin_vel: float = 1.0
    max_ang_vel: float = 1.5
    episode_length: float = 90.0

class Runner:
    def __init__(self, cfg: NavigationCFG):
        self.cfg = cfg
        self.device = gs.device
        self.logs_root = Path("logs") / cfg.exp_name
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        
        gs.init()
        with open(self.logs_root / "cfgs.pkl", "rb") as f:
            env_c, obs_c, rew_c, cmd_c, train_c = pickle.load(f)
            
        env_c["episode_length_s"] = max(env_c.get("episode_length_s", 20.0), cfg.episode_length)
        env_c.update({"termination_if_roll_greater_than": 180.0, "termination_if_pitch_greater_than": 180.0})
        
        self.env = Go2MazeEnv(
            num_envs=1, env_cfg=env_c, obs_cfg=obs_c, reward_cfg=rew_c, command_cfg=cmd_c,
            maze_layout=MAZE, cell_size=cfg.maze_scale, wall_height=cfg.wall_height,
            sensor_cfg={"enable_lidar": True, "enable_depth": True, "enable_topdown": True, 
                       "camera_res": cfg.camera_res, "camera_height": cfg.camera_height},
            show_viewer=False,
        )
        
        runner = OnPolicyRunner(self.env, train_c, str(self.logs_root), device=self.device)
        runner.load(str(self.logs_root / f"model_{cfg.ckpt}.pt"))
        self.policy = runner.get_inference_policy(device=self.device)
        self.lidar_data, self.depth_data = [], []

    def _get_cmd(self, curr_pos, curr_yaw, target):
        diff = target - curr_pos
        yaw_diff = (np.arctan2(diff[1], diff[0]) - curr_yaw + np.pi) % (2 * np.pi) - np.pi
        c, s = np.cos(curr_yaw), np.sin(curr_yaw)
        dx, dy = c * diff[0] + s * diff[1], -s * diff[0] + c * diff[1]
        return (np.clip(self.cfg.lin_gain * dx, -self.cfg.max_lin_vel, self.cfg.max_lin_vel),
                np.clip(self.cfg.lin_gain * dy, -self.cfg.max_lin_vel, self.cfg.max_lin_vel),
                np.clip(self.cfg.ang_gain * yaw_diff, -self.cfg.max_ang_vel, self.cfg.max_ang_vel))

    def run(self):
        obs, _ = self.env.reset()
        if self.env.top_cam: self.env.top_cam.start_recording()
        
        wp_idx, success = 1, False
        for step in range(self.cfg.max_steps):
            if wp_idx >= len(self.env.world_path):
                success = True
                break
                
            pos = self.env.base_pos[0].cpu().numpy()
            target = np.array(self.env.world_path[wp_idx])
            
            if np.linalg.norm(target - pos[:2]) < self.cfg.waypoint_tol:
                print(f"Reached waypoint {wp_idx}")
                wp_idx += 1
                continue
                
            vx, vy, wz = self._get_cmd(pos[:2], self.env.base_euler[0, 2].item(), target)
            self.env.set_command(vx, vy, wz)
            
            with torch.no_grad(): obs, _, _, _ = self.env.step(self.policy(obs))
            
            if self.env.lidar:
                self.lidar_data.append(torch.as_tensor(self.env.lidar.read().points, device=self.device).cpu().numpy().reshape(-1, 3))
            if self.env.depth_cam:
                self.depth_data.append(self.env.depth_cam.read_image().cpu().numpy())
            if self.env.top_cam:
                self.env.top_cam.set_pose(pos=(float(pos[0]), float(pos[1]), self.cfg.camera_height),
                                        lookat=(float(pos[0]), float(pos[1]), 0.0))
                self.env.top_cam.render()

        print(f"Run {'SUCCESS' if success else 'FAILED'}")
        if self.env.top_cam:
            self.env.top_cam.stop_recording(save_to_filename=str(self.cfg.output_dir / "maze_topdown.mp4"))
            
        if self.lidar_data:
            arr = np.stack(self.lidar_data)
            np.savez_compressed(self.cfg.output_dir / "maze_lidar_data.npz", lidar=arr, lidar_data=arr)
        if self.depth_data:
            arr = np.stack(self.depth_data)
            np.savez_compressed(self.cfg.output_dir / "maze_depth_data.npz", depth=arr, 
                              depth_image_data=np.array([d.astype(np.float32) for d in self.depth_data], dtype=object))

if __name__ == "__main__":
    Runner(tyro.cli(NavigationCFG)).run()
