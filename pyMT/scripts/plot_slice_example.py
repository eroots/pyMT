import pyMT.data_structures as DS
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
import pyMT.utils as utils
import matplotlib.pyplot as plt
from pyMT.e_colours import colourmaps as cm

# Read in the model
model = DS.Model('/path/to/model')
# Reads in the data - only needed if you want true model coordinates and/or to plot data locations
data = DS.RawData('/path/to/data/list')
cmap = cm.get_cmap('turbo_r', N=32)
# Convert the model to UTM coordinates - comment out if you want it to be centered at (0,0)
model.origin = data.origin
model.to_UTM()
n_interp = 200 # Number of points to interpolate through
# Convert cell edges to cell centers
x = utils.edge2center(model.dx)
y = utils.edge2center(model.dy)
z = utils.edge2center(model.dz)
# Generate the points you want to interpolate at
qx = np.linspace(x[15], x[-15], n_interp) # A straight line from the 15th cell to the 15th last cell
qy = np.array([y[40] for ii in range(n_interp)]) # zeros all the way for Y (so its a straight N-S line)
# List of Z values is taken directly from the model cell locations, so no interpolation in the Z-direction
# Could modify this if you wanted to.
qz = z[:-15] # All Z-coordinates except for the last 15

query_points = np.zeros((len(qx) * len(qz), 3))
# Loop to generate the full list of 'query' points
# Note that we switch qx and qy here, since the interpolator expects them to correspond to the usual x/y axes
# and not MT-style 'X=north, Y=east' axes.
cc = 0
for ix in range(len(qx)):
    for iz in qz:
        query_points[cc, :] = np.array((qy[ix], qx[ix], iz))
        cc += 1
# Set up the interpolator

interpolator = RGI((y, x, z), np.transpose(model.vals, [1, 0, 2]), bounds_error=False)
vals = interpolator(query_points)
vals = np.reshape(vals, [len(qx), len(qz)])
# Plot things.
plt.pcolormesh(qx, qz, np.log10(vals.T), cmap=cmap, vmin=1, vmax=5)
plt.gca().invert_yaxis()
plt.gca().set_aspect(1)
plt.show()