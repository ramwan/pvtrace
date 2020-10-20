import time
import sys
import functools
import numpy as np
import trimesh
from pvtrace import *

world = Node(
    name="world (air)",
    geometry=Sphere(
        radius=100.0,
        material=Material(refractive_index=1.0),
    )
)

'''
sphere = Node(
    name="sphere (glass)",
    geometry=Sphere(
        radius=1.0,
        material=Material(refractive_index=1.5),
    ),
    parent=world
)
sphere.location = (0, 0, 2)
'''

external_mesh = trimesh.load_mesh(
            '~/Downloads/Temperature_Tower_Generic/files/TempTower.stl',
            file_type='stl')
external_mesh.process(True)
external_mesh.apply_transform([[1/10, 0, 0, 0],[0, 1/10, 0, 0], [0, 0, 1/10, 0], [0, 0, 0, 1/10]])
tower = Node(
    name="mesh",
    geometry=Mesh(
        trimesh=external_mesh,
        material=Material(refractive_index=1.5),
    ),
    parent=world
)
tower.translate((0,0,4))


light = Node(
    name="Light (555nm)",
    light=Light(direction=functools.partial(cone, np.pi/8)),
    parent=world
)

renderer = MeshcatRenderer(wireframe=True, open_browser=True)
scene = Scene(world)
renderer.render(scene)

for ray in scene.emit(100):
    steps = photon_tracer.follow(scene, ray)
    path, events = zip(*steps)
    renderer.add_ray_path(path)
    time.sleep(0.1)


# Wait for Ctrl-C to terminate the script; keep the window open
print("Ctrl-C to close")
while True:
    try:
        time.sleep(.3)
    except KeyboardInterrupt:
        sys.exit()
