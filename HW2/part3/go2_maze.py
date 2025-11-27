from go2_env import Go2Env
import genesis as gs
import numpy as np

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
    # TODO: add class variable and function to implement this class
    
    def build_terrain(self):
        # TODO: complete the build_terrain method to set up the maze with walls, start, and goal positions