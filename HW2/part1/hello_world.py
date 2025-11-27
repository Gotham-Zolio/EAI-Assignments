"""
Assignment: Hello World (Part 1)
Course: Embodied AI 2025 Fall

Objective:
    In this exercise, you will learn the basics of creating and running 
    a simulation scene using the Genesis framework.
    You will:
        - Build a table with a tabletop and legs.
        - Load multiple robots and models into the scene.
        - Set their positions and orientations correctly.
        - Add a camera and optionally record a short simulation.
    Complete all TODO sections marked throughout the code.
"""

import genesis as gs
import dataclasses
import typing
import tyro
import numpy as np


@dataclasses.dataclass
class Args:
    backend: typing.Literal["cpu", "cuda"] = "cpu"
    fixed: bool = False
    show_viewer: bool = True


def add_table(scene: gs.Scene, table_pos: tuple[float, float, float], tabletop_size: tuple[float, float, float], leg_size: tuple[float, float, float]):
    table_x, table_y, table_z = table_pos
    tabletop_dx, tabletop_dy, tabletop_dz = tabletop_size
    leg_dx, leg_dy, leg_dz = leg_size

    tablelegs = []
    for i, j in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        # TODO: compute leg_center_x, leg_center_y, leg_center_z based on table position and leg size
        leg_center_x = table_x + i * (tabletop_dx - leg_dx) / 2
        leg_center_y = table_y + j * (tabletop_dy - leg_dy) / 2
        leg_center_z = table_z + leg_dz / 2
        leg = scene.add_entity(
            gs.morphs.Box(
                # TODO: set size=leg_size
                size=leg_size,
                # TODO: set pos=(leg_center_x, leg_center_y, leg_center_z)
                pos=(leg_center_x, leg_center_y, leg_center_z),
                fixed=True,
            )
        )
        tablelegs.append(leg)

    # TODO: compute tabletop_center_z based on leg height and tabletop thickness
    tabletop_center_z = table_z + leg_dz + tabletop_dz / 2
    tabletop = scene.add_entity(
        gs.morphs.Box(
            size=tabletop_size,
            # TODO: set pos=(table_x, table_y, tabletop_center_z)
            pos=(table_x, table_y, tabletop_center_z),
            fixed=True,
        )
    )

    return tabletop, tablelegs


def main(args: Args):
    gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu)
    scene = gs.Scene(show_viewer=args.show_viewer)
    plane = scene.add_entity(gs.morphs.Plane())

    # TODO: load quadruped robot (Go2)
    # Requirement: place it near the table and make it face the table center.
    go2 = scene.add_entity(
        gs.morphs.URDF(
            # TODO: set file path for Go2 URDF
            file="urdf\\go2\\urdf\\go2.urdf",
            fixed=args.fixed,
            # TODO: set pos near table
            pos=(0, -1, 0.42),
            # TODO: set euler so that it faces the table center
            euler=(0, 0, 90),
        )
    )
    # TODO: define go2_default_qpos

    # TODO: load humanoid robot (G1)
    # Requirement: place it near the table and make it face the table center.
    g1 = scene.add_entity(
        gs.morphs.URDF(
            # TODO: set file path for G1 URDF
            file="urdf\\g1\\g1_29dof_lock_waist_with_hand_rev_1_0.urdf",
            fixed=args.fixed,
            # TODO: set pos near table
            pos=(-0.7, 0, 0.85),
            # TODO: set euler so that it faces the table center
            euler=(0, 0, 0),
        )
    )
    # TODO: define g1_default_qpos

    # TODO: load mobile manipulator robot (Aloha)
    # Requirement: place it near the table and make it face the table center.
    aloha = scene.add_entity(
        gs.morphs.URDF(
            # TODO: set file path for Aloha URDF
            file="urdf\\aloha\\aloha_new.urdf",
            fixed=args.fixed,
            # TODO: set pos near table
            pos=(0, 1, 0.1),
            # TODO: set euler so that it faces the table center
            euler=(0, 0, -90),
        )
    )
    # TODO: define aloha_default_qpos

    table = add_table(
        scene=scene,
        table_pos=(0, 0, 0),
        tabletop_size=(1, 1, 0.05),
        leg_size=(0.05, 0.05, 0.8),
    )

    # TODO: load robotic arm (Franka)
    # Requirement: place it on one side of the table, facing the table center.
    franka = scene.add_entity(
        gs.morphs.URDF(
            # TODO: set file="assets/Robots/panda/panda_v3.urdf"
            file="assets\\Robots\\panda\\panda.urdf",
            fixed=True,
            # TODO: set pos near the table edge (e.g. x=0, y≈0.6, z≈0.85)
            pos=(-0.2, 0, 0.85),
            # TODO: set euler to face the table center
            euler=(0, 0, 0),
        )
    )
    # TODO: define franka_default_qpos

    # TODO: load airplane model
    # Requirement: airplane should face the +x axis direction.
    airplane = scene.add_entity(
        gs.morphs.Mesh(
            file="assets/models/Airplane/airplane.obj",
            # TODO: set scale
            scale=0.003,
            # TODO: set pos above the table
            pos=(0.2, -0.2, 0.9),
            # TODO: set euler so that it faces +x direction
            euler=(90, 0, 90),
            fixed=False,
            collision=True,
        )
    )

    # TODO: load duck model
    # Requirement: duck should face the +x axis direction.
    duck = scene.add_entity(
        gs.morphs.Mesh(
            file="assets/models/duck/duck.obj",
            # TODO: set scale
            scale=0.0003,
            # TODO: set pos above the table
            pos=(0.2, 0.2, 0.9),
            # TODO: set euler so that it faces +x direction
            euler=(90, 0, 90),
            fixed=False,
            collision=True,
        )
    )

    # TODO: create a camera with custom parameters
    cam = scene.add_camera(
        # TODO: set cam params
        res=(960, 720),
        fov=60,
        near=0.1,
        far=10.0,
    )

    record_length = 800
    record_step = 0

    scene.build()

    go2_default_qpos = go2.get_qpos().clone()
    g1_default_qpos = g1.get_qpos().clone()
    aloha_default_qpos = aloha.get_qpos().clone()
    franka_default_qpos = franka.get_qpos().clone()

    robots = [go2, g1, aloha, franka]
    # TODO: define robots_default_qpos = [go2_default_qpos, g1_default_qpos, aloha_default_qpos, franka_default_qpos]
    robots_default_qpos = [go2_default_qpos, g1_default_qpos, aloha_default_qpos, franka_default_qpos]

    cam.start_recording()
    while True:
        scene.step()
        if args.fixed:
            # TODO: keep robots in default pose using control_dofs_position
            for robot, default_qpos in zip(robots, robots_default_qpos):
                robot.control_dofs_position(default_qpos)
        # TODO: move the camera along a circular path and record frames
        angle = record_step * 0.02
        radius = 2
        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        cam_z = 2.8
        cam.set_pose(pos=(cam_x, cam_y, cam_z), lookat=(0, 0, 0.5))
        record_step += 1
        if record_step >= record_length:
            break
        cam.render()
    cam.stop_recording()
        # Hint: use sin/cos to update camera position and cam.set_pose()

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
