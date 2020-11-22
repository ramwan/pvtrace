import matplotlib.pyplot as plt
import numpy as np
import os
import re

keep = False
xflipped = False
yflipped = False
x = []
y = []
emit_one = []
emit_two = []
emit_three = []
emit_fourplus = []
count = []

axes_to_print = []

directory = "Curved corrugation"
flip = False
error = 0.01

with open('./results.out', 'r') as f:
  for line in f:
    # all the lines we're interested in start with "(""
    if re.match(r'^\(', line):
      l1 = re.sub(r'^\(', '', line.strip())
      l2 = re.sub(r'\)', '', l1)
      l3 = re.sub(r'\s+', ' ', l2)
      l4 = re.split(', ', l3)

      xyz = re.split(r' ', l4[0])

      # because I forgot to print minus signs, we have to jump through some hoops here
      if len(x) > 0 and flip:
        if float(xyz[0]) > x[-1] and xflipped == False:
          xflipped = True
          x = [-i for i in x]
        if float(xyz[1]) > y[-1] and yflipped == False:
          yflipped = True
          y = [-j for j in y]

      # only append coords we havent encoutered, keeping ordering
      # this only impacts when we want to plot a grid.
      if float(xyz[0]) not in x:
        x.append( float(xyz[0]) )
      if float(xyz[1]) not in y:
        y.append( float(xyz[1]) )
      emit_one.append( int(l4[1]) )
      emit_two.append( int(l4[2]) )
      emit_three.append( int(l4[3]) )
      emit_fourplus.append( int(l4[4]) )
      count.append( int(l4[5]) )

  if len(x) != 1:
    axes_to_print.append('x')
  if len(y) != 1:
    axes_to_print.append('y')

fig, ax = plt.subplots()
if len(axes_to_print) == 1:
  # pick which axis we print down
  if axes_to_print[0] == 'x':
    ax.plot(x, count, marker='.')
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('PV cell count')
    plt.grid()
    plt.minorticks_on()
    plt.grid(b=True, which='minor', alpha=0.2)
    plt.title(directory)
    plt.show()
  elif axes_to_print[0] == 'y':
    plt.plot(y, count, marker='.')
    ax.set_xlabel('y coordinate')
    ax.set_ylabel('PV cell count')
    plt.grid()
    plt.minorticks_on()
    plt.grid(b=True, which='minor', alpha=0.2)
    plt.title(directory)
    plt.show()
else: # we want to do a 2d plot
  y = y[:-1] # we have an issue with reading in one extra y value
  Z = np.reshape(count, (len(x), len(y)), order='F')
  plt.contourf(x, y, Z)
  ax.set_xlabel('x coordinate')
  ax.set_ylabel('y coordinate')
  plt.colorbar()
  plt.title(directory)
  plt.show()