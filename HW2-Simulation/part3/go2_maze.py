from go2_env import Go2Env
import genesis as gs
import numpy as np
import torch

MAZE = [
    "########",
    "#OO#OOO#",
    "#OO#O#G#",
    "##OOOO##",
    "###O#OO#",
    "#OOOO#O#",
    "#SO#OOO#",
    "########",
]

class Go2MazeEnv(Go2Env):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, maze_layout, cell_size=1.0, wall_height=1.0, sensor_cfg=None, show_viewer=False):
        self.maze_layout = maze_layout
        self.cell_size = cell_size
        self.wall_height = wall_height
        self.sensor_cfg = sensor_cfg or {}
        
        # Parse maze
        self.rows = len(maze_layout)
        self.cols = len(maze_layout[0])
        self.start_pos = (0, 0)
        self.goal_pos = (0, 0)
        
        for r in range(self.rows):
            for c in range(self.cols):
                if maze_layout[r][c] == 'S':
                    self.start_pos = (r, c)
                elif maze_layout[r][c] == 'G':
                    self.goal_pos = (r, c)
        
        # Update start pos
        start_x = self.start_pos[0] * cell_size + cell_size / 2
        start_y = self.start_pos[1] * cell_size + cell_size / 2
        env_cfg["base_init_pos"] = [start_x, start_y, 0.42]
        
        self.lidar = None
        self.depth_cam = None
        self.top_cam = None

        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer)
        
        self.world_path = self._compute_path_bfs()

    def build_terrain(self):
        half_size = self.cell_size / 2
        half_height = self.wall_height / 2
        
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze_layout[r][c] == '#':
                    x = r * self.cell_size + half_size
                    y = c * self.cell_size + half_size
                    
                    self.scene.add_entity(
                        gs.morphs.Box(
                            pos=(x, y, half_height),
                            size=(self.cell_size, self.cell_size, self.wall_height),
                            fixed=True,
                        )
                    )
        
        # Add sensors before scene build
        self._add_sensors()

    def _add_sensors(self):
        if self.sensor_cfg.get("enable_lidar"):
            self.lidar = self.scene.add_sensor(
                gs.sensors.Lidar(
                    entity_idx=self.robot.idx,
                    link_idx_local=0,
                    pos_offset=(0.0, 0.0, 0.2),
                    pattern=gs.sensors.SphericalPattern(
                        fov=(360, 30),
                        n_points=(512, 16),
                    ),
                )
            )
            
        if self.sensor_cfg.get("enable_depth"):
            self.depth_cam = self.scene.add_sensor(
                gs.sensors.DepthCamera(
                    entity_idx=self.robot.idx,
                    link_idx_local=0,
                    pos_offset=(0.2, 0.0, 0.2),
                    pattern=gs.sensors.DepthCameraPattern(
                        res=self.sensor_cfg.get("camera_res", (160, 120)),
                        fov_horizontal=60,
                    ),
                )
            )
            
        if self.sensor_cfg.get("enable_topdown"):
             self.top_cam = self.scene.add_camera(
                res=self.sensor_cfg.get("camera_res", (1280, 720)),
                fov=60,
                pos=(0, 0, self.sensor_cfg.get("camera_height", 10)),
                lookat=(0, 0, 0),
            )

    def _compute_path_bfs(self):
        queue = [(self.start_pos, [self.start_pos])]
        visited = set([self.start_pos])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            (r, c), path = queue.pop(0)
            if (r, c) == self.goal_pos:
                world_path = []
                for (pr, pc) in path:
                    wx = pr * self.cell_size + self.cell_size / 2
                    wy = pc * self.cell_size + self.cell_size / 2
                    world_path.append((wx, wy))
                return world_path
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.maze_layout[nr][nc] != '#' and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(nr, nc)]))
        return []