import argparse
import os
import pickle
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from go2_env import Go2Env

COMMANDS = [
    ("c1", [1.0, 0.0, 0.0]),
    ("c2", [0.0, 0.5, 0.0]),
    ("c3", [0.0, 0.0, 1.0]),
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="go2-locomotion")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output-dir", type=str, default="part3")
    parser.add_argument("--camera-height", type=float, default=3.0)
    parser.add_argument("--camera-res", type=int, nargs=2, default=(640, 480))
    return parser.parse_args()

class EvalWrapper(Go2Env):
    """Wrapper to inject a camera for evaluation."""
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, cam_res, cam_h, show_viewer=False):
        self.cam_res = tuple(cam_res)
        self.cam_h = float(cam_h)
        self.cam = None
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer)

    def build_terrain(self):
        super().build_terrain()
        # Add camera
        self.cam = self.scene.add_camera(
            res=self.cam_res,
            fov=45.0,
            pos=(0.0, 0.0, self.cam_h),
            lookat=(0.0, 0.0, 0.0),
        )

def plot_tracking(times: np.ndarray, targets: np.ndarray, measured: np.ndarray, filename: str):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["V_x (m/s)", "V_y (m/s)", "W_z (rad/s)"]
    
    for i, ax in enumerate(axes):
        ax.plot(times, targets[:, i], "r--", label="Command", linewidth=2)
        ax.plot(times, measured[:, i], "b-", label="Response", linewidth=2)
        ax.set_ylabel(f"{labels[i]}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
            
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def run_eval(env: EvalWrapper, policy, cmd_val: torch.Tensor, name: str, duration: float, out_dir: Path, cam_h: float):
    obs, _ = env.reset()
    steps = int(duration / env.dt)

    hist_meas = []
    hist_targ = []
    
    cam = env.cam
    cam.start_recording()

    env.set_command(cmd_val[0], cmd_val[1], cmd_val[2])
    
    cmd_np = cmd_val.cpu().numpy()

    for _ in range(steps):
        with torch.no_grad():
            act = policy(obs)
        obs, _, _, _ = env.step(act)
        
        vels = torch.cat([env.base_lin_vel[0, :2], env.base_ang_vel[0, 2:3]])
        hist_meas.append(vels.detach().cpu().numpy())
        hist_targ.append(cmd_np)
        
        pos = env.base_pos[0].detach().cpu().numpy()
        cam.set_pose(
            pos=(float(pos[0]), float(pos[1]), cam_h),
            lookat=(float(pos[0]), float(pos[1]), 0.0)
        )
        cam.render()
    video_path = out_dir / f"eval_{name}_topdown.mp4"
    cam.stop_recording(save_to_filename=str(video_path))
    print(f"Saved video to {video_path}")

    t_axis = np.arange(len(hist_meas)) * env.dt
    plot_path = out_dir / f"eval_{name}_plot.png"
    plot_tracking(
        t_axis, 
        np.array(hist_targ), 
        np.array(hist_meas), 
        str(plot_path)
    )
    print(f"Saved plot to {plot_path}")

def main():
    args = get_args()
    gs.init()

    log_dir = Path("logs") / args.exp_name
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = log_dir / "cfgs.pkl"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
        
    with open(cfg_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    env = EvalWrapper(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        cam_res=args.camera_res,
        cam_h=args.camera_height,
        show_viewer=False
    )

    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.load(str(log_dir / f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    for name, cmd_list in COMMANDS:
        print(f"\nRunning Case: {name} -> {cmd_list}")
        t_cmd = torch.tensor(cmd_list, device=env.device, dtype=gs.tc_float)
        
        run_eval(
            env=env,
            policy=policy,
            cmd_val=t_cmd,
            name=name,
            duration=args.duration,
            out_dir=out_dir,
            cam_h=args.camera_height
        )

if __name__ == "__main__":
    main()
