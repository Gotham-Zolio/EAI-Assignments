"""
Assignment: Sensors – IMU (Part 2)
Course: Embodied AI 2025 Fall

Objective:
    In this exercise, you will attach an IMU sensor to the Franka arm’s
    end-effector, move the robot in a circular path, and record the IMU
    readings for both measured and ground-truth acceleration.
    You will:
        - Attach the IMU to the robot’s end-effector.
        - Implement circular motion control for the end-effector.
        - Read IMU data during simulation.
        - Visualize and compare measured vs. ground-truth acceleration.
    Complete all TODO sections marked throughout the code.
"""

import genesis as gs
import numpy as np
import tyro
import dataclasses
import typing
import matplotlib.pyplot as plt


@dataclasses.dataclass
class Args:
    """Command-line arguments for the IMU simulation."""
    backend: typing.Literal["cpu", "cuda"] = "cpu"
    show_viewer: bool = True
    steps: int = 1000
    circle_radius: float = 0.15


def main(args: Args):
    # ============================================================
    # Scene Setup
    # ============================================================
    gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=args.show_viewer,
    )

    # Add plane and Franka robot
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.URDF(file="assets/Robots/panda/panda.urdf", fixed=True))
    
    end_effector = franka.get_link("panda_link7")
    motors_dof = np.arange(7)

    # ============================================================
    # TODO: Attach IMU sensor to the end-effector
    # ============================================================
    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=franka.idx,
            link_idx_local=end_effector.idx_local,
            pos_offset=(0, 0, 0),
            euler_offset=(0, 0, 0),
            # sensor characteristics
            acc_axes_skew=(0.0, 0.1, 0.2),
            acc_bias=(0.3, 0.4, 0.5),
            acc_noise=(0.1, 0.1, 0.1),
            gyro_noise=(0.1, 0.1, 0.1),
        )
    )

    # ============================================================
    # Build Scene and Configure Controller
    # ============================================================
    scene.build()
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))

    circle_center = np.array([0.4, 0.0, 0.5])
    rate = 10 / 180 * np.pi  # angular speed in rad/step

    # ============================================================
    # TODO: Implement circular motion control
    # ============================================================
    def control_franka_circle_path(i):
        phi = i * rate
        pos = circle_center + np.array([
            args.circle_radius * np.cos(phi),
            args.circle_radius * np.sin(phi),
            0.0
        ]) 
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos,
            quat=np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[motors_dof], motors_dof)

    # ============================================================
    # Simulation Loop and Data Recording
    # ============================================================
    acc_measured, acc_gt = [], []

    for i in range(args.steps):
        control_franka_circle_path(i)
        scene.step()

        # TODO: Read and record IMU data (both measured & ground truth)
        acc_measured.append(imu.read().lin_acc.cpu().numpy())
        acc_gt.append(imu.read_ground_truth().lin_acc.cpu().numpy())

    acc_measured = np.array(acc_measured)
    acc_gt = np.array(acc_gt)

    # ============================================================
    # TODO: Visualize IMU data using matplotlib
    # ============================================================
    t = np.arange(args.steps) * scene.sim_options.dt
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes = ['X', 'Y', 'Z']
    
    for i in range(3):
        axs[i].plot(t, acc_measured[:, i], label='measured', alpha=0.7)
        axs[i].plot(t, acc_gt[:, i], label='ground truth', linestyle='--', linewidth=2)
        
        axs[i].set_ylabel(f'Acc {axes[i]} (m/s²)')
        axs[i].legend()
        axs[i].grid(True)
        axs[i].set_title(f'Acceleration - {axes[i]}')
    
    axs[2].set_xlabel('Time (s)')
    plt.tight_layout()
    
    plt.savefig('imu_acceleration_plot.png')
    if args.show_viewer:
        plt.show()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
