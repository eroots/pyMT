import urllib
import csv
import codecs
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import cmocean
import pyMT.data_structures as DS
import pyMT.utils as utils
import pickle


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
#   - bath_out: File to save bathymetry data to. The script goes online and downloads the required data, so you can use this and bath_file to make sure you're only downloading it once.
#   - model_out: File to save the modified model to. The only difference between this and 'model_file' is that the relevant model cells will have air resistivities.
#   - cov_out: File to save the covariance file to. This file is required by ModEM when using topography / bathymetry.
#   - data_out: File to save the modified data to. The only difference between this and 'data_file' is that the data here will have non-zero elevation values.
# There are a few other things you can change if you want below these lines. Comment in / out as needed.

def get_bathymetry(minlat, maxlat, minlon, maxlon, stride=1):
    print('Retrieving topo data from latitude {:>6.3g} to {:>6.3g}, longitude {:>6.3g} to {:>6.3g}'.format(minlat, maxlat, minlon, maxlon))
    # Read data from: http://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v6.html
    response = urllib.request.urlopen('http://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v6.csv?topo[(' \
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
    # LANAI
    # model_file = 'E:/phd/Nextcloud/data/Regions/Lanai/test/test.model'
    # list_file = 'E:/phd/Nextcloud/data/Regions/Lanai/j2/lanai_good_only.lst'
    # data_file = 'E:/phd/Nextcloud/data/Regions/Lanai/test/lanai_test_Z.dat'
    # bath_file = 'E:/phd/Nextcloud/data/Regions/Lanai/test/bathy.p'
    # bath_out = []
    # model_out = 'E:/phd/Nextcloud/data/Regions/Lanai/test/test_wTopoAndOcean.model'
    # cov_out = 'E:/phd/Nextcloud/data/Regions/Lanai/test/lanai_wTopoAndOcean.cov'
    # data_out = 'E:/phd/Nextcloud/data/Regions/Lanai/test/lanai_wTopoAndOcean_Z.dat'
    #################################
    # COREDILLA
    # model_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/jim_topo_test/mm/mm/inv2/test_topo.model'
    # list_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/jim_topo_test/mm/mm/inv2/j2/select.lst'
    # data_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/jim_topo_test/mm/mm/inv2/modem.dat'
    # # bath_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/jim_topo_test/mm/mm/inv2/topo.xyz'
    # bath_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/jim_topo_test/mm/mm/inv2/topo.p'
    # # bath_out = 'C:/Users/eroots/phd/ownCloud/data/Regions/jim_topo_test/mm/mm/inv2/topo.p'
    # model_out = 'C:/Users/eroots/phd/ownCloud/data/Regions/jim_topo_test/mm/mm/inv2/topo/test_wTopo.model'
    # cov_out = 'C:/Users/eroots/phd/ownCloud/data/Regions/jim_topo_test/mm/mm/inv2/topo/lanai_topo.cov'
    # data_out = 'C:/Users/eroots/phd/ownCloud/data/Regions/jim_topo_test/mm/mm/inv2/topo/lanai__topoTest_Z.dat'
    ################################
    # WESTERN SUPERIOR
    size = 'nest'
    model_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/full/wst_fullmantle_hs500_{}.model'.format(size)
    list_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_mantle_fullrun_ffmt.lst'
    data_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/full/wst_fullmantle_LAMBERT_all_flagged.dat'
    data_out = []
    # bath_file = []
    bath_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/bathy_{}.p'.format(size)
    bath_out = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/bathy_{}.p'.format(size)
    model_out = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/full/wst_hs500_wOcean_{}.model'.format(size)
    cov_out = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/full/wst_hs500_wOcean_{}.cov'.format(size)
    ################################
    #  Ciomadual
    # model_file = 'E:/phd/Nextcloud/data/Regions/Ciomadul/cio5/1D/smoothed/topo/cio1D-smooth.model'
    # model_file = 'E:/phd/Nextcloud/data/Regions/Ciomadul/cio5/1D/smoothed/topo/cio1D_nest.model'
    # list_file = 'E:/phd/Nextcloud/data/Regions/Ciomadul/j2/originals/rotated/fixed/ffmt_sorted.lst'
    # data_file = 'E:/phd/Nextcloud/data/Regions/Ciomadul/cio5/1D/smoothed/topo/cio4_halfPers.dat'
    # bath_file = 'E:/phd/Nextcloud/data/Regions/Ciomadul/cio5/1D/smoothed/topo/bathy_nest.p'
    # bath_file = []
    # bath_out = 'E:/phd/Nextcloud/data/Regions/Ciomadul/cio5/1D/smoothed/topo/bathy_nest.p'
    # model_out = 'E:/phd/Nextcloud/data/Regions/Ciomadul/cio5/1D/smoothed/topo/cioHS_wTopoAndOcean_nest.model'
    # cov_out = 'E:/phd/Nextcloud/data/Regions/Ciomadul/cio5/1D/smoothed/topo/cioHS_wTopoAndOceanFlipX_nest.cov'
    # data_out = 'E:/phd/Nextcloud/data/Regions/Ciomadul/cio5/1D/smoothed/topo/cioHS_wTopoAndOcean_nest.dat'
    ###############################
    raw_data = DS.RawData(list_file)
    data = DS.Data(listfile=list_file, datafile=data_file)
    model = DS.Model(model_file)
    model.origin = raw_data.origin
    lat_pad = 5
    lon_pad = 5
    data_collect_stride = 4
    with_topography = True
    cmap = cmocean.cm.haline
    ####################################
    # If you want to modify the vertical meshing, do it now (see examples below)
    # Add 20 layers that are each 200 m thick, then append the existing mesh (I used this for testing purposes)
    # Z = [50] * 20 + model.zCS
    # model.zCS = Z
    # model.vals[:,:,:] = 10
    # model.zCS = [50] * 12 + [20] + model.zCS
    # model.dz = list(np.arange(0, 620, 20)) + list(np.logspace(np.log10(620), 4.5, 80)) + list(np.logspace(4.5, 6, 20))[1:]
    # model.zCS = [20] * 31 + list(np.logspace(np.log10(20), 3, 80)) + list(np.logspace(3.1, 5, 20))
    # for ii in range(20):
        # model.dz_insert(0, 50)
    # Another testing mesh, this time with 100 m layers from 0-10 km, then 1 km layers from 10-100 km depth
    # model.dz = list(range(0, 10000, 100)) + list(range(10000, 100000, 1000))
    ####################################
    # Change as needed
    # model.background_resistivity = 100
    # model.generate_half_space()
    # This one is needed to make sure the projection to lat/long is correct.
    # model.UTM_zone = '4Q'
    # model.UTM_zone = '35N'
    # model.UTM_zone = '16N'
    model.UTM_zone = '16U'
    model.to_latlong()
    minlat, maxlat = model.dx[0] - lat_pad, model.dx[-1] + lat_pad
    minlon, maxlon = model.dy[0] - lon_pad, model.dy[-1] + lon_pad
    # minlat = 18
    # maxlat = 22
    # minlon = -157
    # maxlon = -154
    if bath_file:
        try:
            bathy = pickle.load(open(bath_file, 'rb'))
            lat, lon, topo = bathy[0, :], bathy[1, :], bathy[2, :]
        except pickle.UnpicklingError:
            bathy = np.genfromtxt(bath_file)
            lat, lon, topo = bathy[:, 1], bathy[:, 0], bathy[:, 2]
    else:
        lat, lon, topo = get_bathymetry(minlat, maxlat, minlon, maxlon, stride=data_collect_stride)
        bathy = np.array((lat, lon, topo))
        pickle.dump(bathy, open(bath_out, 'wb'))
    
    grid_x, grid_y, grid_z = bathymetry_to_model(model, lat, lon, topo)
    # insert_topography(model, grid_x, grid_y, grid_z)
    insert_oceans(model, grid_x, grid_y, grid_z, with_topography=with_topography)
    model.to_local()
    reposition_data(data, model)
    # # Should be straightforward to assign model vals and covariances for AIR
    # # However there is the additional complication that the data file must be altered so that sites are located properly
    # # From ModEM Manual: Z = 0 is the highest point in the topography, all site Z coords must be specified relative to this, with +ve Z being downward (sites should have +ve Z)
    model.write(model_out)
    model.set_exceptions()
    model.write_covariance(cov_out)
    if data_out:
        data.write(data_out, use_elevation=True)
    raw_data.locations = raw_data.get_locs(mode='latlong')
    plot_it(grid_x, grid_y, grid_z, raw_data.locations, cmap=cmap)
    # main()

