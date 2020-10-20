from pvtrace import LSC, CustomLSC, circular_mask, cone, lambertian
import trimesh
import time
import sys
import numpy as np
import functools

n1 = 1.5
external_mesh = trimesh.load_mesh(
            '~/Documents/pvtrace/STL-geometry/Twist Meshes/180 Degree.STL',
            file_type='stl')
external_mesh.process(True)
zscale = trimesh.transformations.scale_matrix(1/5, direction=[0, 0, 1])
xscale = trimesh.transformations.scale_matrix(1/5, direction=[1, 0, 0])
yscale = trimesh.transformations.scale_matrix(1/5, direction=[0, 1, 0])
rotation = trimesh.transformations.rotation_matrix(np.radians(45), direction=[0, 1, 0])
transform = trimesh.transformations.concatenate_matrices(xscale, yscale, zscale, rotation)
external_mesh.apply_transform(transform)

_, extents = trimesh.bounds.oriented_bounds(external_mesh)
print("Bounding box dimensions: ", extents)

num_sims = 300
custom_lsc = CustomLSC(external_mesh, extents) # mesh size is arbitrary rn
custom_lsc.add_light(
  "Light1",
  (0, 0, extents[2]*0.5), # some portion of the z height of our bounding box
  (np.radians(180), (1, 0, 0)),
  functools.partial(cone, np.radians(20)),
  #lambertian,
  None, # defaults to 555nm i think, but we can specify a distribution
  functools.partial(circular_mask, 0.3))

custom_lsc.show(open_browser=True, max_history=num_sims*10)
custom_lsc.simulate(num_sims)
#custom_lsc.report()

# Wait for Ctrl-C to terminate the script; keep the window open
# print("Ctrl-C to close")
# while True:
#     try:
#         time.sleep(.3)
#     except KeyboardInterrupt:
#         sys.exit()