import urllib
import csv
import codecs
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import cmocean
import pyMT.data_structures as DS
import pyMT.utils as utils
import pickle
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
# from skimage.transform import downscale_local_mean

#####################################
########## INSTRUCTIONS #############
# Run script through command line (python bath2model.py) or in ipython (run bath2model) 
# Also note that this script may have dependencies not required by the rest of pyMT (e.g., urllib, codecs, Basemap).
# If you get any 'module not found' errors, use anaconda to install them then try again.
# I've also run into issues with Basemap. If you get an error 'KeyError: 'PROJ_LIB'', try running 'conda install -c conda-forge proj4'.
# This script should be the last thing that is run to generate your inversion inputs (I can't guarantee that mesh_designer and data_plot will play nice and preserve the changes made here)
# Replace file names below with your own file names (Lines 128-135)
#   - model_file: input model file (the one you want to add topography to). This should be set up as you want it, minus topography.
#          i.e., it should have the desired resistivities and lateral meshing, as well as the sufficient vertical mesh to support topography at your required resolution.
#   - list_file: The list file (as used by pyMT) containing the EDI or j-format files to read in.
#   - data_file: The ModEM data file to modify. As with the model file, this should already be set up as you want it for the inversion - this script will only modify the elevations.
#   - bath_file: The file containing the bathymetry / topography data. If this is the first time running this script, set bath_file = []. The script will generate the bathymetry and save it to bath_out
#                If your bathymetry is in a Geotiff file, set this to the path. Note that you have to have rasterio installed for this to work.
#   - bath_out: File to save bathymetry data to. The script goes online and downloads the required data, so you can use this and bath_file to make sure you're only downloading it once.
#   - model_out: File to save the modified model to. The only difference between this and 'model_file' is that the relevant model cells will have air resistivities.
#   - cov_out: File to save the covariance file to. This file is required by ModEM when using topography / bathymetry.
#   - data_out: File to save the modified data to. The only difference between this and 'data_file' is that the data here will have non-zero elevation values.
# There are a few other things you can change if you want below these lines. Comment in / out as needed.
# Lines you'll need to change start at line #142
def get_bathymetry(minlat, maxlat, minlon, maxlon, stride=1, resource=30):
    print('Retrieving topo data from latitude {:>6.3g} to {:>6.3g}, longitude {:>6.3g} to {:>6.3g}'.format(minlat, maxlat, minlon, maxlon))
    # Read data from: http://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v6.html
    # Had to change to v1 since v6 stopped working for some reason...
    # 'https://coastwatch.pfeg.noaa.gov/erddap/griddap/srtm15plus.csv?z%5B(59):1:(63)%5D%5B(-141):1:(-137)%5D'
    if resource == 15:
        response = urllib.request.urlopen('https://coastwatch.pfeg.noaa.gov/erddap/griddap/srtm15plus.csv?z%5B(' \
                                      +str(minlat)+'):'+str(stride)+':('+str(maxlat)+')%5D%5B('+str(minlon)+'):'+str(stride)+':('+str(maxlon)+')%5D')
    elif resource == 30:
        response = urllib.request.urlopen('http://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v1.csv?topo[(' \
                                      +str(maxlat)+'):'+str(stride)+':('+str(minlat)+')][('+str(minlon)+'):'+str(stride)+':('+str(maxlon)+')]')
    else:
        print('Resource should be 15 or 30. Using 30.')
        response = urllib.request.urlopen('http://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v1.csv?topo[(' \
                                      +str(maxlat)+'):'+str(stride)+':('+str(minlat)+')][('+str(minlon)+'):'+str(stride)+':('+str(maxlon)+')]')
    r = csv.reader(codecs.iterdecode(response, 'utf-8'))
    # Initialize variables
    lat, lon, topo = [], [], []
    data = list(r)
    for row in data[2:]:
        # try:
        lat.append(float(row[0]))
        lon.append(float(row[1]))
        topo.append(float(row[2]))
    # Convert into numpy arrays
    lat = np.array(lat, dtype='float')
    lon = np.array(lon, dtype='float')
    topo = np.array(topo, dtype='float')
    return lat, lon, topo


def bathymetry_to_model(model, lat, lon, topo):
    print('Gridding topo data...')

    grid_x, grid_y = np.meshgrid(model.dy, model.dx)
    grid_z = scipy.interpolate.griddata((lon, lat), topo, (grid_x, grid_y), method='nearest')
    return grid_x, grid_y, grid_z


def insert_oceans(model, grid_x, grid_y, grid_z, with_topography=False):
    print('Inserting oceans into model...')
    center_x = utils.edge2center(model.dx)
    center_y = utils.edge2center(model.dy)
    if with_topography:
        highest_point = np.max(grid_z)
    else:
        highest_point = np.float64(0)
    sea_level_idx = np.argmin(abs(model.dz - highest_point))
    for ix, xx in enumerate(center_x):
        for iy, yy in enumerate(center_y):
            # for iz, zz in enumerate(model.dz):
            if grid_z[ix, iy] < 0:
                iz = np.argmin(abs(model.dz + grid_z[ix, iy] - highest_point))
                model.vals[ix, iy, sea_level_idx:iz] = model.RHO_OCEAN


def insert_topography(model, grid_x, grid_y, grid_z):
    print('Inserting mountains into model...')
    center_x = utils.edge2center(model.dx)
    center_y = utils.edge2center(model.dy)
    highest_point = np.max(grid_z)
    for ix, xx in enumerate(center_x):
        for iy, yy in enumerate(center_y):
            # for iz, zz in enumerate(model.dz):
            # if grid_z[ix, iy] > 0:
                iz = np.argmin(abs(-highest_point + model.dz + grid_z[ix, iy]))
                model.vals[ix, iy, :iz] = model.RHO_AIR


def reposition_data(data, model):
    for ii, site in enumerate(data.site_names):
        x, y = data.locations[ii, 0], data.locations[ii, 1]
        center_x = utils.edge2center(model.dx)
        center_y = utils.edge2center(model.dy)
        center_z = utils.edge2center(model.dz)
        cell_x = np.argmin(abs(x - center_x))
        cell_y = np.argmin(abs(y - center_y))
        site_z = [z for iz, z in enumerate(model.dz[:-1]) if model.vals[cell_x, cell_y, iz] != model.RHO_AIR][0]
        data.sites[site].locations['elev'] = site_z


def plot_it(grid_x, grid_y, grid_z, locations=None, cmap=None):
    print('Plotting stuff...')
    # Create map
    minlat, maxlat = np.min(grid_y.ravel()), np.max(grid_y.ravel())
    minlon, maxlon = np.min(grid_x.ravel()), np.max(grid_x.ravel())
    # m = Basemap(projection='mill', llcrnrlat=minlat, urcrnrlat=maxlat,
                # llcrnrlon=minlon, urcrnrlon=maxlon, resolution='l')
    # x, y = m(grid_x, grid_y)
    # fig1 = plt.figure()
    if not cmap:
        cmap = cmocean.cm.delta
    plt.pcolor(grid_x, grid_y, grid_z, cmap=cmap)
    # m.pcolor(x, y, grid_z, cmap=cmap, zorder=0)#, vmin=-2000, vmax=2000)
    # m.drawcoastlines()
    # m.drawmapboundary()
    if locations is not None:
        plt.plot(locations[:, 1], locations[:, 0], 'kv', zorder=5)
    plt.title('SMRT30 - Bathymetry/Topography')
    cbar = plt.colorbar(orientation='vertical', extend='both')
    cbar.ax.set_xlabel('meters')
    plt.show()
# Save figure (without 'white' borders)
# plt.savefig('topo.png', bbox_inches='tight')


if __name__ == '__main__':
    # Define the domain of interest
    #################################
    model_file = '/file/location/here'
    list_file  = '/file/location/here'
    data_file  = '/file/location/here'
    bath_file  = '/file/location/here'
    bath_out   = '/file/location/here'
    model_out  = '/file/location/here'
    cov_out    = '/file/location/here'
    data_out   = '/file/location/here'
    # Resample topography file. Mainly useful to resample from GeoTiff.
    # Set to False if you don't want it, otherwise set to a 2x1 tuple, e.g., (2, 2) will resample every 2nd cell in each direction
    resample_topo = False 
    lat_pad = 2  # Extend the download of topo/bathymetry data by this many degrees in latitude
    lon_pad = 2  # Extend the download of topo/bathymetry data by this many degrees in longitude
    data_collect_stride = 1 # Integer multiple of the resolution desired (base resolution is 30 arc seconds) - change this if you have a big model or you'll download way more data than is needed!
    with_topography = True  # Include topography?
    with_oceans = True      # Include oceans?
    resource = 30           # Which resource to use (15 or 30 arc seconds)
    UTM_zone = '7N' # Set to the UTM zone of your data
    # cmap = cmocean.cm.haline
    cmap = None
    raw_data = DS.RawData(list_file)
    data = DS.Data(listfile=list_file, datafile=data_file)
    model = DS.Model(model_file)
    model.origin = raw_data.origin
    model.UTM_zone = UTM_zone  # Set your UTM zone
    # If you want to modify the vertical meshing, do it now (see examples below)
    # I find its best to have constant layer thickness inside the topography
    # Example below we use 25x50 m layers (1250 m total), then append the original model thicknesses starting at the 24th layer (i.e., where the thickness is ~50 m)
    Z = [50] * 33 + model.zCS[24:]
    model.zCS = Z
    # Some other test possibilities for setting up your model + mesh below
    ####################################    
    # Set up the z-mesh as a series logspace intervals
    # model.dz = list(np.arange(0, 620, 20)) + list(np.logspace(np.log10(620), 4.5, 80)) + list(np.logspace(4.5, 6, 20))[1:]
    # Reset the resistivities to a half-space
    # model.background_resistivity = 100 
    # model.generate_half_space()
    #####################################
    # Set the z-mesh as 31x20 m cells followed by a logspace setup
    # model.zCS = [20] * 31 + list(np.logspace(np.log10(20), 3, 80)) + list(np.logspace(3.1, 5, 20))
    #####################################
    # Another testing mesh, this time with 100 m layers from 0-10 km, then 1 km layers from 10-100 km depth
    # model.dz = list(range(0, 10000, 100)) + list(range(10000, 100000, 1000))
    ####################################
    # Change as needed
    # model.background_resistivity = 100
    # model.generate_half_space()
    # This one is needed to make sure the projection to lat/long is correct.
    #####################################
    # You can change these if you want - they set what the lat/long bounds of your topography are
    # Be mindful of how you set this up - you can run out of RAM if you're not careful.
    model.to_latlong()
    minlat, maxlat = np.round(model.dx[0] - lat_pad, decimals=2), np.round(model.dx[-1] + lat_pad, decimals=2)
    minlon, maxlon = np.round(model.dy[0] - lon_pad, decimals=2), np.round(model.dy[-1] + lon_pad, decimals=2)
    # minlat = 18
    # maxlat = 22
    # minlon = -157
    # maxlon = -154
    if bath_file:
        if bath_file.endswith('tif') or bath_file.endswith('.tiff'):
            print('Reading topography from GeoTIFF {}'.format(bath_file))
            # This is a huge workaround until I figure out how to do it properly.
            dst_crs = 'EPSG:4326'  # WGS
            with rasterio.open(bath_file,) as src:
                transform, width, height = calculate_default_transform(src.crs,
                                                                       'EPSG:4326',
                                                                       src.width,
                                                                       src.height,
                                                                       *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({'crs': dst_crs,
                                'transform':transform,
                                'width': width,
                                'height':height})
            
                with rasterio.open('temp.tif', 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(source=rasterio.band(src,i),
                        destination=rasterio.band(dst,i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_resolution=0.01,
                        dst_crs=dst_crs)
            with rasterio.open('temp.tif') as dst:
                topo = dst.read(1).ravel()
                lat, lon = [], []
                for iy in range(dst.height):
                    for ix in range(dst.width):
                        x, y = dst.transform * (ix, iy)
                        lat.append(y)
                        lon.append(x)
                lat = np.array(lat)
                lon = np.array(lon)
        else:

            try:
                bathy = pickle.load(open(bath_file, 'rb'))
                lat, lon, topo = bathy[0, :], bathy[1, :], bathy[2, :]
            except pickle.UnpicklingError:
                bathy = np.genfromtxt(bath_file)
                lat, lon, topo = bathy[:, 1], bathy[:, 0], bathy[:, 2]
    else:
        lat, lon, topo = get_bathymetry(minlat, maxlat, minlon, maxlon, stride=data_collect_stride, resource=resource)
        with open(bath_out, 'wb') as f:
            pickle.dump(np.array((lat, lon, topo)), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    if resample_topo:
        print('Resampling grid first...')
        nx = len(np.unique(lat))
        ny = len(np.unique(lon))
        im = np.reshape(topo, [nx, ny])
        lat = np.reshape(lat, [nx, ny])[::resample_topo[0], ::resample_topo[0]]
        lon = np.reshape(lon, [nx, ny])[::resample_topo[0], ::resample_topo[0]]
        lat, lon = lat.flatten(), lon.flatten()
        im = downscale_local_mean(im, resample_topo)
        topo = im.flatten()

    grid_x, grid_y, grid_z = bathymetry_to_model(model, lat, lon, topo)
    if with_topography:
        insert_topography(model, grid_x, grid_y, grid_z)
    if with_oceans:
        insert_oceans(model, grid_x, grid_y, grid_z, with_topography=with_topography)
    model.to_local()
    reposition_data(data, model)
    model.write(model_out)
    model.set_exceptions()
    model.write_covariance(cov_out)
    if data_out:
        data.write(data_out, use_elevation=True)
    raw_data.locations = raw_data.get_locs(mode='latlong')
    plot_it(grid_x, grid_y, grid_z, raw_data.locations, cmap=cmap)
    # main()

