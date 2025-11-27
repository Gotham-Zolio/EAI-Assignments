"""
Assignment: Sensors – LiDAR (Part 2.2)
Course: Embodied AI 2025 Fall

Objective:
    In this exercise, you will build a simple maze using box primitives,
    place a mobile robot in the maze, and navigate it from the start (S)
    to the goal (G) using velocity control. You will attach LiDAR and
    depth sensors to the robot to record environmental perception data.

    You will:
        - Construct the maze scene and find a path from S to G.
        - Attach LiDAR and depth sensors to the robot.
        - Control the robot’s linear and angular velocities to follow
          the path.
        - Record and save LiDAR scans and depth images.

    Complete all TODO sections marked throughout the code.
"""

import dataclasses
import typing
import pathlib
import numpy as np
import tyro
import genesis as gs
from genesis.utils.geom import euler_to_quat
from collections import deque


@dataclasses.dataclass
class Args:
    backend: typing.Literal["cpu", "cuda"] = "cpu"
    show_viewer: bool = True
    maze_scale: float = 1.0
    output: pathlib.Path = pathlib.Path("part2")


MAZE = [
    "##G##",
    "#OOO#",
    "###O#",
    "#SOO#",
    "#####",
]


def find_start_goal(maze):
    start = goal = None
    for r, row in enumerate(maze):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    return start, goal


def bfs_path(maze, start, goal):
    H, W = len(maze), len(maze[0])
    queue = deque([(start, [start])])
    visited = {start}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                if maze[nr][nc] != "#":
                    queue.append(((nr, nc), path + [(nr, nc)]))
                    visited.add((nr, nc))
    raise ValueError("No path found from S to G.")


def grid_to_world(r, c, maze, scale):
    H, W = len(maze), len(maze[0])
    x = (c - W / 2) * scale
    y = ((H / 2) - r) * scale
    return x, y


def main(args: Args):
    gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(gravity=(0, 0, -9.8), dt=0.01),
        show_viewer=args.show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())

    wall_h = 1.0
    for r, row in enumerate(MAZE):
        for c, ch in enumerate(row):
            if ch == "#":
                x, y = grid_to_world(r, c, MAZE, args.maze_scale)
                scene.add_entity(
                    gs.morphs.Box(
                        size=(args.maze_scale, args.maze_scale, wall_h),
                        pos=(x, y, wall_h / 2),
                        fixed=True,
                    )
                )

    start, goal = find_start_goal(MAZE)
    path = bfs_path(MAZE, start, goal)
    world_path = [grid_to_world(r, c, MAZE, args.maze_scale) for r, c in path]

    sx, sy = grid_to_world(*start, MAZE, args.maze_scale)
    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(sx, sy, 0.3))
    )

    # ============================================================
    # TODO: Add LiDAR and Depth sensors
    # ============================================================

    lidar = scene.add_sensor(
        gs.sensors.Lidar(
            entity_idx=robot.idx,
            link_idx_local=0,  # base link
            pos_offset=(0, 0, 0.2),
            pattern=gs.sensors.SphericalPattern(
                fov=(360, 30),
                n_points=(512, 16),
            ),
        )
    )
    depth = scene.add_sensor(
        gs.sensors.DepthCamera(
            entity_idx=robot.idx,
            link_idx_local=0,  # base link
            pos_offset=(0.2, 0, 0.2),
            pattern=gs.sensors.DepthCameraPattern(
                res=(160, 120),
                fov_horizontal=60,
            ),
        )
    )

    scene.build()

    lidar_data = []
    depth_image_data = []

    pos = np.array([sx, sy, 0.3], dtype=np.float32)
    yaw = 0.0
    v_lin = 0.8
    w_max = 1.5
    dt = scene.dt

    for target_x, target_y in world_path[1:]:
        while True:
            delta = np.array([target_x - pos[0], target_y - pos[1]])
            dist = np.linalg.norm(delta)
            if dist < 0.05:
                break

            target_yaw = np.arctan2(delta[1], delta[0])
            yaw_err = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi

            w = np.clip(yaw_err / dt, -w_max, w_max)
            v = v_lin * (abs(yaw_err) < 1e-2 and 1.0 or 0.0)

            yaw += w * dt
            pos[0] += v * np.cos(yaw) * dt
            pos[1] += v * np.sin(yaw) * dt

            robot.set_pos(pos)
            robot.set_quat(euler_to_quat((0, 0, yaw * 180 / np.pi)))

            # ====================================================
            # TODO: Step simulation and collect sensor readings
            # ====================================================
            scene.step()
            
            l_points = lidar.read().points.cpu().numpy()
            lidar_data.append(l_points.reshape(-1, 3))
            
            d_map = depth.read().distances.cpu().numpy()
            depth_image_data.append(d_map)

    # ============================================================
    # TODO: Save LiDAR and Depth data
    # ============================================================
    args.output.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        args.output / "lidar_maze_data.npz",
        lidar_data=np.array(lidar_data, dtype=object)
    )
    np.savez(
        args.output / "depth_maze_data.npz",
        depth_image_data=np.array(depth_image_data, dtype=object)
    )
    print(f"Simulation finished. Data saved to {args.output}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
