import urllib
from io import BytesIO
import csv
import codecs
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pickle

# Definine the domain of interest
minlat = 18
maxlat = 22
minlon = -157
maxlon = -154

# Read data from: http://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v6.html
response = urllib.request.urlopen('http://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v6.csv?topo[(' \
                                  +str(maxlat)+'):1:('+str(minlat)+')][('+str(minlon)+'):1:('+str(maxlon)+')]')

# data = BytesIO(response.read())

# r = csv.DictReader(data,dialect=csv.Sniffer().sniff(data.read(1000)))
# data.seek(0)
# r = csv.reader(response.read().decode('utf-8'))
r = csv.reader(codecs.iterdecode(response, 'utf-8'))

# Initialize variables
lat, lon, topo = [], [], []

# Loop to parse 'data' into our variables
# Note that the second row has the units (i.e. not numbers). Thus we implement a
# try/except instance to prevent the loop for breaking in the second row (ugly fix)
data = list(r)
for row in data[2:]:
    # try:
    lat.append(float(row[0]))
    lon.append(float(row[1]))
    topo.append(float(row[2]))
    # except:
        # print('Row '+str(row)+' is a bad...')

# Convert 'lists' into 'numpy arrays'
lat  = np.array(lat,  dtype='float')
lon  = np.array(lon,  dtype='float')
topo = np.array(topo, dtype='float')

# Data resolution determined from here:
# http://coastwatch.pfeg.noaa.gov/erddap/info/usgsCeSrtm30v6/index.html
resolution = 0.008333333333333333

# Determine the number of grid points in the x and y directions
nx = complex(0,(max(lon)-min(lon))/resolution)
ny = complex(0,(max(lat)-min(lat))/resolution)

# Build 2 grids: One with lats and the other with lons
grid_x, grid_y = np.mgrid[min(lon):max(lon):nx,min(lat):max(lat):ny]

# Interpolate topo into a grid (x by y dimesions)
grid_z = scipy.interpolate.griddata((lon,lat),topo,(grid_x,grid_y),method='linear')

# Make an empty 'dictionary'... place the 3 grids in it.
TOPO = {}
TOPO['lats']=grid_y
TOPO['lons']=grid_x
TOPO['topo']=grid_z

# Save (i.e. pickle) the data for later use
# This saves the variable TOPO (with all its contents) into the file: topo.p
pickle.dump(TOPO, open('topo.p','wb'))

# Create map
m = Basemap(projection='mill', llcrnrlat=minlat,urcrnrlat=maxlat,llcrnrlon=minlon, urcrnrlon=maxlon,resolution='l')
x,y = m(grid_x,grid_y)

fig1 = plt.figure()
cs = m.pcolor(x,y,grid_z,cmap=plt.cm.jet)
m.drawcoastlines()
m.drawmapboundary()
plt.title('SMRT30 - Bathymetry/Topography')
cbar = plt.colorbar(orientation='horizontal', extend='both')
cbar.ax.set_xlabel('meters')

# Save figure (without 'white' borders)
plt.savefig('topo.png', bbox_inches='tight')