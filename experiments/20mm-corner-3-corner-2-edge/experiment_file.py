from pvtrace import LSC, CustomLSC, circular_mask, cone, lambertian, Light
import trimesh
import time
import sys
import numpy as np
import functools
import re

#name = "30mm-Radius-Corner"
name = "20mm Radius - 3 Corner Strips & Both Edge PV Cells"

lscmesh = trimesh.load(
      "./"+name+".3MF",
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

#rays_per_light = 20000
n = 1.5 # refractive index of the material
custom_lsc = CustomLSC(lscmesh, extents) 

'''
HOW TO USE THIS SO FAR:

1. run the program with "*"_points set to a small number to check the
   positioning and orientation of the mesh. This will let you determine
   where x_start and y_start should be.
'''

num_lights = 0 # value will be updated, leave it at 0
rays_per_light = 10000 # recommended to use at least 10000 for statistical accuracy
x_points = 60 # min: 1
y_points = 1 # min: 1
x_start = extents[0] / 5
x_step = extents[0] / x_points
y_start = 0
y_step = 0

for x in range(x_points):
  for y in range(y_points):
    num_lights += 1
    xcoord = x_start + x_step * x
    ycoord = y_start + y_step * y
    coords = (xcoord, ycoord, extents[2]*0.5)
    custom_lsc.add_light(
      str(coords),
      coords,
      rotation=(np.radians(180), (1, 0, 0)),
      direction=functools.partial(cone, np.radians(0.001)),
      wavelength=None,
      position=None)

'''
This section is for when you want to have the light shining along
the x-axis. We only need to worry about z changes here so z-coord
related things are in this section only.

Uncomment below when you want to use it and comment the above out.
'''

'''
z_points = 60
z_start = -extents[2] * 0.075
z_step = (extents[2] * 0.035 - z_start) / z_points
for z in range(z_points):
  for y in range(y_points):
    num_lights += 1
    ycoord = y_start + y_step * y
    zcoord = z_start + z_step * z
    coords = (x_start, ycoord, zcoord)
    custom_lsc.add_light(
      str(coords),
      coords,
      rotation=(np.radians(90), (0, 1, 0)),
      direction=functools.partial(cone, np.radians(0.001)),
      wavelength=None,
      position=None)
'''

#custom_lsc.show(open_browser=True, max_history=rays_per_light*num_lights*10)
#custom_lsc.show(open_browser=False, max_history=1)

# when we simulate, we cycle through the lights so we want to emit
# num_lights * rays_per_light times
custom_lsc.simulate(rays_per_light * num_lights, update_vis=False)
custom_lsc.summary()