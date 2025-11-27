import genesis as gs
import numpy as np

gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=False)
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 0, 0.3))
)

print("Adding sensors...")
try:
    lidar = scene.add_sensor(
        gs.sensors.Lidar(
            entity_idx=robot.idx,
            link_idx_local=0,
            pos_offset=(0, 0, 0.2),
            pattern=gs.sensors.SphericalPattern(
                fov=(360, 30),
                n_points=(100, 16), # Low res for test
            )
        )
    )
    
    depth = scene.add_sensor(
        gs.sensors.DepthCamera(
            entity_idx=robot.idx,
            link_idx_local=0,
            pos_offset=(0.2, 0, 0.2),
            pattern=gs.sensors.DepthCameraPattern(
                res=(64, 48), # Low res for test
                fov_horizontal=60,
            )
        )
    )
    
    scene.build()
    scene.step()
    
    print("Reading Lidar...")
    l_data = lidar.read()
    print(f"Lidar data keys: {dir(l_data)}")
    # Check for points or rays
    if hasattr(l_data, 'points'):
        print(f"Lidar points shape: {l_data.points.shape}")
    if hasattr(l_data, 'rays'):
        print(f"Lidar rays shape: {l_data.rays.shape}")
    if hasattr(l_data, 'dist'):
        print(f"Lidar dist shape: {l_data.dist.shape}")

    print("Reading Depth...")
    d_data = depth.read()
    print(f"Depth data keys: {dir(d_data)}")
    if hasattr(d_data, 'depth'):
        print(f"Depth map shape: {d_data.depth.shape}")
    if hasattr(d_data, 'points'):
        print(f"Depth points shape: {d_data.points.shape}")

except Exception as e:
    print(f"Error: {e}")

