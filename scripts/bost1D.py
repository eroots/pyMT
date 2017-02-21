import pyMT.data_structures as WSDS
import numpy as np
import pyMT.utils as utils
import sys
# from scipy.interpolate import griddata
# import matplotlib.pyplot as plt


def do_the_bosticks(listfile=None, datafile=None, comp=None,
                    N_interp=None, z_bounds=None, filter_width=1, sites_slice=None):
    if listfile is None:
        print('Listfile is None')
        listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\j2\allsites.lst'
    if datafile is None:
        print('datafile is None')
        datafile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\j2\allsites_1.data'
    if listfile:
        raw = WSDS.RawData(listfile=listfile)
    else:
        raw = WSDS.Data(datafile=datafile)
    if not sites_slice:
        sites_slice = ('p91011', 'p91009', 'e90005', 'e90001', 'g90003', 'p91003',
                       'p91006', 'p91006', 'p91008', 'p91012', 'e89001', 'e90002',
                       'g90004', 'g90002', 'g90006', 'p91013', 'p91014', 'p91015',
                       'p91016', 'p91017', 'p91019')
    data = WSDS.Data(listfile=listfile,
                     datafile=datafile)
    NS = len(raw.site_names)
    if not N_interp:
        NZ_interp = 200
        NXY_interp = 200
    elif len(N_interp) == 1:
        NZ_interp = N_interp
        NXY_interp = N_interp
    else:
        NZ_interp = N_interp[0]
        NXY_interp = N_interp[1]
    locs = np.zeros((NS, 2))
    all_depths = []
    all_bosticks = []
    for ii, site_name in enumerate(raw.site_names):
        site = raw.sites[site_name]
        locs[ii, 1] = data.sites[site.name].locations['X'] / 1000
        locs[ii, 0] = data.sites[site.name].locations['Y'] / 1000
        # locs[ii, 1] = raw.sites[site.name].locations['Long']
        # locs[ii, 0] = raw.sites[site.name].locations['Lat']
        bostick, depth = utils.compute_bost1D(site, comp=comp, filter_width=filter_width)[:2]
        if any(np.isnan(depth)):
            print(site_name)
            sys.exit()
        all_depths.append(depth)
        all_bosticks.append(bostick)
    all_depths_flat = np.array(utils.flatten_list(all_bosticks))
    data_points = np.zeros((len(all_depths_flat), 3))
    c = 0
    for ii, ds in enumerate(all_depths):
        for d in ds:
            data_points[c, 0] = locs[ii, 0]
            data_points[c, 1] = locs[ii, 1]
            data_points[c, 2] = d
            c += 1
    try:
        y = [data.sites[site].locations['X'] / 1000 for site in sites_slice]  # if site in data.site_names]
        x = [data.sites[site].locations['Y'] / 1000 for site in sites_slice]  # if site in data.site_names]
        # y = [raw.sites[site].locations['Long'] for site in sites_slice]  # if site in data.site_names]
        # x = [raw.sites[site].locations['Lat'] for site in sites_slice]  # if site in data.site_names]
    except KeyError as e:
        if (type(sites_slice[0]) == str):
            print('Site {} not found. Check your site list.'.format(e))
            print(e)
            sys.exit()
        else:
            x = sites_slice[:int(len(sites_slice) / 2)]
            y = sites_slice[int(len(sites_slice) / 2):]
    # This sorting is south-north. Might not make sense for west-east profiles.
    x, y = ([y2 for (x2, y2) in sorted(zip(y, x))],
            [x2 for (x2, y2) in sorted(zip(y, x))])
    print(x, y)
    print(raw.site_names)
    print(data.periods)
    print('Data file is {}'.format(datafile))
    print('List file is {}'.format(listfile))
    V = np.array(utils.flatten_list(all_bosticks))

    if not z_bounds:
        z_bounds = (0.1, 300)
    zi = np.logspace(np.log10(z_bounds[0]), np.log10(z_bounds[1]), NZ_interp)
    yi = np.linspace(min(y), max(y), NXY_interp)
    xi = np.interp(yi, y, x)
    X = np.ndarray.flatten(np.tile(xi, (NZ_interp, 1)), order='F')
    Y = np.ndarray.flatten(np.tile(yi, (NZ_interp, 1)), order='F')
    Z = np.ndarray.flatten(np.tile(zi, (1, NXY_interp)), order='F')
    query_points = np.transpose(np.array((X, Y, Z)))
    return data_points, query_points, V
    # mod = griddata(data_points, V, query_points, method='linear')
    # griddata
    # new_bosticks = np.zeros((NZ_interp, NS))
    # for ii, bost in enumerate(all_bosticks):
    #     new_bosticks[:, ii] = np.interp(new_depths, all_depths[ii], bost)

    # bostick_model = np.zeros((NXY_interp, NXY_interp, NZ_interp))
    # [X, Y] = np.meshgrid(np.linspace(min(locs[:, 0]), max(locs[:, 0]), NXY_interp),
    #                      np.linspace(min(locs[:, 1]), max(locs[:, 1]), NXY_interp))
    # for ii in range(NZ_interp):
    #     bostick_model[:, :, ii] = griddata(locs, new_bosticks[ii, :], (X, Y), 'cubic')


# if __name__ == '__main__':
    # data, query, values = do_the_bosticks()
