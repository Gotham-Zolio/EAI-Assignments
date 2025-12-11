
import genesis as gs
gs.init()
scene = gs.Scene()
plane = scene.add_entity(gs.morphs.Plane())
depth = scene.add_sensor(
    gs.sensors.DepthCamera(
        entity_idx=plane.idx,
        pos_offset=(0.2, 0, 0.2),
        pattern=gs.sensors.DepthCameraPattern(
            res=(160, 120),
            fov_horizontal=60,
        ),
    )
)
scene.build()
print(dir(depth))
