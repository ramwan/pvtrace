from pvtrace import LSC, CustomLSC, circular_mask, cone, lambertian, Light
import trimesh
import time
import sys
import numpy as np
import functools
import re

name = "10mm-Radius-Corner"

lscmesh = trimesh.load(
      "/Users/raymondwan/Documents/pvtrace/3MF-geometry/"+name+".3mf",
      file_type='3mf',
      force='mesh')

#external_mesh.process(True)
zscale = trimesh.transformations.scale_matrix(1/3, direction=[0, 0, 1])
xscale = trimesh.transformations.scale_matrix(1/3, direction=[1, 0, 0])
yscale = trimesh.transformations.scale_matrix(1/3, direction=[0, 1, 0])
#rotation = trimesh.transformations.rotation_matrix(np.radians(180), direction=[0, 0, 1])
transform = trimesh.transformations.concatenate_matrices(zscale, xscale, yscale)
lscmesh.apply_transform(transform)

# find the first solar cell face and print the surface normal vector
print("Solar cell face normal -> ")
for i in range( len(lscmesh.visual.face_colors) ):
  face = lscmesh.visual.face_colors[i]
  if face[0] == 255 and face[1] == 0 and face[2] == 0:
    print(lscmesh.face_normals[i])
    break

_, extents = trimesh.bounds.oriented_bounds(lscmesh)
print("Bounding box -> ")
print(extents)

num_sims = 2
n = 1.5 # refractive index of the material
custom_lsc = CustomLSC(lscmesh, extents) 

custom_lsc.add_light(
  "Point1",
  (0, 0, extents[2]*0.5),
  rotation=(np.radians(180), (1, 0, 0)),
  direction=functools.partial(cone,np.radians(0.001)),
  wavelength=None,
  position=None)
custom_lsc.add_light(
    "Point2",
    (10, 0, extents[2]*0.5),
    rotation=(np.radians(180), (1, 0, 0)),
    direction=functools.partial(cone,np.radians(0.001)),
    wavelength=None,
    position=None)
custom_lsc.show(open_browser=True, max_history=num_sims*10)

# when we simulate, we cycle through the lights so we want to emit
# num_lights * rays_per_light times
custom_lsc.simulate(num_sims)
custom_lsc.report()