import numpy as np
import pyMT.data_structures as WSDS
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pyMT.e_colours import colourmaps as cm
import pyMT.utils as utils
import pyMT.gplot as gplot
from copy import deepcopy
from scipy.spatial.distance import euclidean


cmap = cm.jet()


if __name__ == '__main__':
    # local_path = 'C:/Users/eroots'
    local_path = 'E:/phd/Nextcloud/'
    # filename = local_path + 'data/Regions/MetalEarth/AG/AG_plotset.dat'
    # listfile = local_path + 'data/Regions/MetalEarth/j2/upper_abitibi_hex.lst'
    # out_path = local_path + 'Documents/ME_transects/Upper_Abitibi/Paper/RoughFigures/PT/phi2_betaBack/betaCircle/'
    filename = local_path + 'data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wst_cullmantle3_LAMBERT_ZK_flagged.dat'
    listfile = local_path + 'data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst'
    respfile = local_path + 'data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wstZK_lastIter.dat'
    out_path = 'E:/phd/NextCloud/Documents/ME_Transects/wst/PTs/by_period/ind_only/response/'
    # filename = local_path + 'data/Regions/snorcle/j2/2020-collation-ian/grid_north.lst'
    # listfile = local_path + 'data/Regions/snorcle/j2/2020-collation-ian/grid_north.lst'
    # out_path = local_path + 'Documents/ME_transects/Upper_Abitibi/Paper/RoughFigures/PT/phi2_betaBack/betaCircle/'
    # jpg_file_name = local_path + 'ArcMap/AG/cio_georeferenced.jpg'
    jpg_file_name = 'E:/phd/NextCloud/data/ArcMap/WST/WSBoundaries_Lambert_wMCR.jpg'
    out_file = 'wst_inds_'
    ext = ['.png', '.svg']
    dpi = 150
    padding = 20
    save_fig = 1
    bostick_depth = None
    cutoff_distance = 3500
    remove_close_sites = 0
    arrow_types = ['R']
    data_type = ['response']
    normalize = False
    data = WSDS.Data(filename, listfile=listfile)
    raw = WSDS.RawData(listfile)
    response = WSDS.Response(respfile)
    # data = deepcopy(raw)
    # data.locations = rawdata.get_locs(mode='latlong')
    freq_skip = 0

    all_sites = deepcopy(data.site_names)
    # Remove redunantly close points
    if remove_close_sites:
        for ii, site1 in enumerate(data.site_names):
            for jj, site2 in enumerate(data.site_names):
                dist = euclidean((data.locations[ii, 1], data.locations[ii, 0]),
                                 (data.locations[jj, 1], data.locations[jj, 0]))
                if dist < cutoff_distance and site1 in all_sites and (site1 != site2):
                    if site2 in all_sites:
                        all_sites.remove(site2)
        rm_sites = [site for site in data.site_names if site not in all_sites]
        # rm_sites = [site for site in data.site_names[2:]]
        data.remove_sites(sites=rm_sites)
        raw.remove_sites(sites=rm_sites)
        response.remove_sites(sites=rm_sites)
    raw.locations = raw.get_locs(mode='lambert')
    data.locations = raw.locations
    response.locations = raw.locations
    data.reset_errors() # Just to unflag the BB stations so their induction arrows are included
    
    # for ii in range(len(raw.locations)):
    #     lon, lat = utils.project((raw.locations[ii, 1], raw.locations[ii, 0]), zone=17, letter='U')[2:]
    #     raw.locations[ii, 1], raw.locations[ii, 0] = lon, lat
    # data.locations = raw.locations / 1000
    
    if jpg_file_name:
        im = plt.imread(jpg_file_name)
        with open(jpg_file_name[:-3] + 'jgw', 'r') as f:
            xsize = float(f.readline())
            dummy = f.readline()
            dummy = f.readline()
            ysize = 1 * float(f.readline())
            x1 = float(f.readline())
            y2 = float(f.readline())
        x2 = x1 + xsize * im.shape[1]
        y1 = y2 + ysize * im.shape[0]
        extents = [x1, x2, y1, y2]

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    MV = gplot.MapView(fig=fig)
    MV.window['figure'] = fig
    MV.window['axes'] = [ax]
    MV.colourmap = 'greys_r'
    MV.site_data['data'] = data
    MV.site_data['raw_data'] = raw
    MV.site_data['response'] = response
    MV.site_names = data.site_names
    MV.padding_scale = 10
    MV.induction_scale = 5
    MV.induction_cutoff = 1
    MV.arrow_colours = {'raw_data': {'R': 'b', 'I': 'r'},
                        'data': {'R': 'r', 'I': 'k'},
                        'response': {'R': 'g', 'I': 'c'}}
    # # MV.site_locations['generic'] = MV.get_locations(sites=MV.generic_sites)
    # MV.site_locations['active'] = MV.get_locations(
    #     sites=MV.active_sites)
    MV.site_locations['all'] = data.locations
    first_time = 0
    # for ii in range(len(data.periods)):
    for ii in range(data.NP):
    # for ii in range(0, len(data.periods), freq_skip + 1):
    # for ii in [0]:
    # for ii in range(len(data.periods)):   
        if not first_time:
            MV.window['figure'] = plt.figure(figsize=(16, 10))
            MV.window['axes'] = [MV.window['figure'].add_subplot(111)]
        else:
            first_time = 1
        # period = data.sites[data.site_names[0]].periods[ii]
        period = data.periods[ii]
        # if period < 1:
        #     period = -1 / period
        period = str(int(period))
        if jpg_file_name:
            MV.plot_image(im, extents)
        MV.plot_induction_arrows(data_type=data_type, normalize=normalize, period_idx=ii, arrow_type=arrow_types)
        MV.set_axis_limits(bounds=[min(data.locations[:, 1]) - padding,
                                   max(data.locations[:, 1]) + padding,
                                   min(data.locations[:, 0]) - padding,
                                   max(data.locations[:, 0]) + padding])
        MV.window['axes'][0].set_aspect(1)
        MV.window['axes'][0].set_xlabel('Easting (m)', fontsize=14)
        MV.window['axes'][0].set_ylabel('Northing (m)', fontsize=14)
        MV.window['axes'][0].set_title('Period: {0:.5g} s'.format(data.sites[data.site_names[0]].periods[ii]))
        # MV.window['axes'][0].set_title('NB Depth: {0:.5g} s'.format(bostick_depth))
        # ells, vals, norm_vals = plot_ellipse(data, fill_param='phi_max')
        if save_fig:
            for file_format in ext:
                plt.savefig(out_path + out_file + 'idx' + str(ii) + '_p' + period + file_format, dpi=dpi,
                            transparent=True)
            plt.close('all')
            MV.window['colorbar'] = None
            # ax.clear()

            # for x in caxes:
            #     x.clear()
        else:
            plt.show()
