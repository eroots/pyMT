import urllib
import csv
import codecs
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pyMT.data_structures as DS
import pyMT.utils as utils
import pickle


def get_bathymetry(minlat, maxlat, minlon, maxlon):
    print('Retrieving topo data from latitude {:>6.3g} to {:>6.3g}, longitude {:>6.3g} to {:>6.3g}'.format(minlat, maxlat, minlon, maxlon))
    # Read data from: http://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v6.html
    response = urllib.request.urlopen('http://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v6.csv?topo[(' \
                                  +str(maxlat)+'):1:('+str(minlat)+')][('+str(minlon)+'):1:('+str(maxlon)+')]')

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
    m = Basemap(projection='mill', llcrnrlat=minlat, urcrnrlat=maxlat,
                llcrnrlon=minlon, urcrnrlon=maxlon, resolution='l')
    x, y = m(grid_x, grid_y)
    # fig1 = plt.figure()
    if not cmap:
        cmap = plt.cm.jet
    m.pcolor(x, y, grid_z, cmap=cmap)
    m.drawcoastlines()
    m.drawmapboundary()
    if locations is not None:
        m.plot(locations[:, 1], locations[:, 0], 'kv')
    plt.title('SMRT30 - Bathymetry/Topography')
    cbar = plt.colorbar(orientation='horizontal', extend='both')
    cbar.ax.set_xlabel('meters')
    plt.show()
# Save figure (without 'white' borders)
# plt.savefig('topo.png', bbox_inches='tight')


if __name__ == '__main__':
    # Define the domain of interest
    #################################
    # LANAI
    model_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/Lanai/test/test.model'
    list_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/Lanai/j2/lanai_good_only.lst'
    data_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/Lanai/test/lanai_test_Z.dat'
    bath_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/Lanai/test/bathy.p'
    model_out = 'C:/Users/eroots/phd/ownCloud/data/Regions/Lanai/test/test_wTopoAndOcean.model'
    cov_out = 'C:/Users/eroots/phd/ownCloud/data/Regions/Lanai/test/lanai_wTopoAndOcean.cov'
    data_out = 'C:/Users/eroots/phd/ownCloud/data/Regions/Lanai/test/lanai_wTopoAndOcean_Z.dat'
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
    # model_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_usarray_nested.model'
    # list_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/wst/j2/ME_wst_usarray.lst'
    # bath_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/wst/bathy.p'
    # data_out = []
    # # data_file = 
    # # # bath_file = []
    # # bath_out = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/wst/cull1/bg1000_wOcean/bathy.p'
    # model_out = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_usarray_nested_wOcean.model'
    # cov_out = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_usarray_nested_ocean.cov'
    raw_data = DS.RawData(list_file)
    data = DS.Data(listfile=list_file, datafile=data_file)
    model = DS.Model(model_file)
    model.zCS = [200] * 20 + model.zCS
    # model.dz = list(range(0, 10000, 100)) + list(range(10000, 100000, 1000))  # Just for testing, make an evenly spaced subsurface
    model.background_resistivity = 100
    model.generate_half_space()
    model.origin = raw_data.origin
    model.UTM_zone = '4Q'
    # model.UTM_zone = '16N'
    # model.UTM_zone = '15U'
    model.to_latlong()
    minlat, maxlat = model.dx[0] - 1, model.dx[-1] + 1
    minlon, maxlon = model.dy[0] - 1, model.dy[-1] + 1
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
        lat, lon, topo = get_bathymetry(minlat, maxlat, minlon, maxlon)
        bathy = np.array((lat, lon, topo))
        pickle.dump(bathy, open(bath_out, 'wb'))
    
    grid_x, grid_y, grid_z = bathymetry_to_model(model, lat, lon, topo)
    insert_topography(model, grid_x, grid_y, grid_z)
    insert_oceans(model, grid_x, grid_y, grid_z, with_topography=True)
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
    plot_it(grid_x, grid_y, grid_z, raw_data.locations)
    # main()

