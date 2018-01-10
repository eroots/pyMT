import pyMT.data_structures as WSDS
import matplotlib.pyplot as plt
import shapefile
from pyproj import Proj
import numpy as np
import matplotlib.ticker as plticker


loc = plticker.MultipleLocator(base=75.0)  # this locator puts ticks at regular intervals

abi = WSDS.RawData(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions' +
                   r'\abi-gren\New\j2\allsites.lst')
# wst = WSDS.RawData(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions' +
#                    r'\wst\New\j2\all.lst')
wst = WSDS.RawData(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions' +
                   r'\wst\New\j2\southeastern.lst')
abi.locations = abi.get_locs(mode='latlong')
wst.locations = wst.get_locs(mode='latlong')
MT_stations = ([ix for ix in abi.locations[:, 0]] + [ix for ix in wst.locations[:, 0]],
               [iy for iy in abi.locations[:, 1]] + [iy for iy in wst.locations[:, 1]])
p = Proj(proj='utm', zone=17, ellps='WGS84')
sl = shapefile.Reader('D:/PhD/Seismic Lines/Export_Output_11')
MT_stations = p(MT_stations[1], MT_stations[0])
MT_stations = np.transpose(np.array(MT_stations)) / 1000
sx, sy = ([], [])
for shape in sl.shapeRecords():
    sx += [ix[0] / 1000 for ix in shape.shape.points[:]]
    sy += [iy[1] / 1000 for iy in shape.shape.points[:]]
transect = np.transpose(np.array((sx, sy)))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(MT_stations[:, 0], MT_stations[:, 1], 'r^', mew=2, mec='k', markersize=8)
ax.plot(sx, sy, 'bo', mew=2, mec='k', markersize=6)
ax.xaxis.set_major_locator(loc)
plt.show()
