from pvtrace import LSC, CustomLSC, circular_mask, cone, lambertian
import trimesh
import time
import sys
import numpy as np
import functools
import re

name = "Trapeze-Corri"

print(trimesh.available_formats())
lscmesh = trimesh.load(
      "/Users/raymondwan/Documents/pvtrace/3MF-geometry/"+name+".3mf",
      file_type='3mf',
      force="mesh")

facesDict = {}
colouredFaces = []
interested = False
ids = []
colours = []
with open("/Users/raymondwan/Documents/pvtrace/3MF-geometry/"+name+"/3D/3dmodel.model")\
as f:
  # Not dealing with files that have been new-lined to be
  # human readable.
  for line in f:
    splitline = line.replace("><", ">\n<").split("\n")
    for l in splitline:
      if interested:
        interested = False
        captured = re.search(r'color="(.*?)"', l)
        colours.append(captured.group(1))

      if "<m:colorgroup" in l:
        interested = True
        captured = re.search(r'id="(.*?)"', l)
        ids.append(captured.group(1))

      if "<triangle v1" in l and "pid=" in l:
        v1 = re.search(r'v1="(.*?)"', l).group(1)
        v2 = re.search(r'v2="(.*?)"', l).group(1)
        v3 = re.search(r'v3="(.*?)"', l).group(1)
        colourId = re.search(r'pid="(.*?)"', l).group(1)
        colourCode = ids.index(colourId)
        colouredFaces.append( ( (int(v1), int(v2), int(v3)),\
                                colours[colourCode] ) )

# for some reason our vertices/faces are getting scaled???

i = 0
for (f1, f2, f3) in lscmesh.faces:
  facesDict[i] = (f1, f2, f3)
  i += 1

n1 = 1.5

#external_mesh.process(True)
zscale = trimesh.transformations.scale_matrix(1/5, direction=[0, 0, 1])
xscale = trimesh.transformations.scale_matrix(1/5, direction=[1, 0, 0])
yscale = trimesh.transformations.scale_matrix(1/5, direction=[0, 1, 0])
#rotation = trimesh.transformations.rotation_matrix(np.radians(45), direction=[0, 0, 0])
transform = trimesh.transformations.concatenate_matrices(xscale, yscale, zscale)
lscmesh.apply_transform(transform)

_, extents = trimesh.bounds.oriented_bounds(lscmesh)
print("Bounding box dimensions: ", extents)

num_sims = 100
# mesh size is arbitrary rn
custom_lsc = CustomLSC(lscmesh, extents, \
                       counter_faces=colouredFaces, \
                       faces_dict=facesDict) 
custom_lsc.add_light(
  "Light1",
  (0, 0, extents[2]*0.5), # some portion of the z height of our bounding box
  (np.radians(180), (1, 0, 0)),
  functools.partial(cone, np.radians(0.1)),
  #lambertian,
  None, # defaults to 555nm i think, but we can specify a distribution
  functools.partial(circular_mask, 0.3))

custom_lsc.show(open_browser=False, max_history=1)
custom_lsc.simulate(num_sims)
custom_lsc.report()

# Wait for Ctrl-C to terminate the script; keep the window open
# print("Ctrl-C to close")
# while True:
#     try:
#         time.sleep(.3)
#     except KeyboardInterrupt:
#         sys.exit()