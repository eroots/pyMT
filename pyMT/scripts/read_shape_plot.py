import shapefile
import numpy as np
import matplotlib.pyplot as plt
import pyMT.data_structures as WSDS
import pyMT.utils as utils
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.feature import ShapelyFeature


shp_file_base='C:/Users/eric/phd/ownCloud/data/ArcMap/test2.shp'
list_file = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/j2/culled_allSuperior.lst'

# data = WSDS.RawData(list_file)
data.locations = data.get_locs(mode='latlong')
# dat_dir='../shapefiles/'+shp_file_base +'/'
sf = shapefile.Reader(shp_file_base)
# transform = ccrs.PlateCarree()
transform = ccrs.TransverseMercator(central_longitude=-85, central_latitude=49,
                                    false_northing=5430000, false_easting=645000)
# We want the data plotted in UTM, and we will convert them to UTM before plotting
# transform = ccrs.UTM(zone=16)
for ii in range(len(data.locations)):
    easting, northing = utils.project((data.locations[ii, 1], data.locations[ii, 0]), zone=16, letter='U')[2:]
    data.locations[ii, 1], data.locations[ii, 0] = easting, northing
print('number of shapes imported:', len(sf.shapes()))
print(' ')
print('geometry attributes in each shape:')
for name in dir(sf.shape()):
    if not name.startswith('__'):
        print(name)
# This defines the projection we want to plot in
ax = plt.axes(projection=transform)

# ax.coastlines()

shp = shapereader.Reader(shp_file_base)
# Note I use ccrs.PlateCarree() here because that is the projection the shapefile is in
# I.E., latlong, not UTM. cartopy will take care of converting them as long as these are
# all defined properly.
for record, shape in zip(shp.records(), shp.geometries()):
    ax.add_geometries([shape], ccrs.PlateCarree(), facecolor='lightgrey', edgecolor='black')
# shape_feature = ShapelyFeature(Reader(shp_file_base).geometries(),
#                                transform, edgecolor='black')
# ax.add_feature(shape_feature, facecolor='blue')
# plt.show()
for val, label in zip(ax.get_xticks(), ax.get_xticklabels()):
    label.set_text(str(val))
    label.set_position((val, 0))

for val, label in zip(ax.get_yticks(), ax.get_yticklabels()):
    label.set_text(str(val))
    label.set_position((0, val))

plt.tick_params(bottom=True, top=True, left=True, right=True,
                labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(True)
plt.grid(True)
plt.plot(data.locations[:, 1], data.locations[:, 0], 'k.', transform=transform)
# ax.set_extent([-200000, 1800000, 5100000, 5700000], transform)
plt.show()
# for shape in list(sf.iterShapes()):
#     npoints = len(shape.points)  # total points
#     nparts = len(shape.parts)  # total parts

#     if nparts == 1:
#         x_lon = np.zeros((len(shape.points), 1))
#         y_lat = np.zeros((len(shape.points), 1))
#         easting = np.zeros((len(shape.points), 1))
#         northing = np.zeros((len(shape.points), 1))
#         for ip in range(len(shape.points)):
#             x_lon[ip] = shape.points[ip][0]
#             y_lat[ip] = shape.points[ip][1]
#             easting[ip], northing[ip] = utils.project((x_lon[ip], y_lat[ip]), zone=16, letter='U')[2:]

#         # plt.plot(x_lon, y_lat, 'k', transform=ccrs.PlateCarree())
#         plt.plot(easting, northing, 'k', transform=transform)

#     else:  # loop over parts of each shape, plot separately
#         for ip in range(nparts):  # loop over parts, plot separately
#             i0 = shape.parts[ip]
#             if ip < nparts - 1:
#                 i1 = shape.parts[ip + 1] - 1
#             else:
#                 i1 = npoints
#             seg = shape.points[i0:i1 + 1]
#             x_lon = np.zeros((len(seg), 1))
#             y_lat = np.zeros((len(seg), 1))
#             easting = np.zeros((len(seg), 1))
#             northing = np.zeros((len(seg), 1))
#             for ip in range(len(seg)):
#                 x_lon[ip] = seg[ip][0]
#                 y_lat[ip] = seg[ip][1]
#                 easting[ip], northing[ip] = utils.project((x_lon[ip], y_lat[ip]), zone=16, letter='U')[2:]
# my_coords = [50, -87]
# zoom_scale = 1
# bbox = [my_coords[0]-zoom_scale,my_coords[0]+zoom_scale,
#         my_coords[1]-zoom_scale,my_coords[1]+zoom_scale]
# m = Basemap(transform='merc', llcrnrlat=bbox[0], urcrnrlat=bbox[1],
#             llcrnrlon=bbox[2], urcrnrlon=bbox[3], lat_ts=10,resolution='i')
# x, y = m(easting, northing)
# m.plot(x, y)
# plt.plot(easting, northing, 'k') 
# plt.plot(data.locations[:, 1], data.locations[:, 0], 'k.', transform=transform)
# plt.show()