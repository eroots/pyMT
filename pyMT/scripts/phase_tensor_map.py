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


def normalize_ellipse(phi):
    phi_min = abs(phi.phi_min)
    phi_max = abs(phi.phi_max)
    phi_min, phi_max = utils.normalize([phi_min, phi_max])
    return phi_min, phi_max


def plot_ellipse(data, fill_param):
    ells = []
    # data.locations = data.get_locs(mode='latlong')
    for site_name in data.site_names:
        site = data.sites[site_name]
        jx = np.cos(np.arange(0, 2 * np.pi, np.pi / 30))
        jy = np.sin(np.arange(0, 2 * np.pi, np.pi / 30))
        phi_x = site.phase_tensors[-6].phi[1, 1] * jx + site.phase_tensors[-6].phi[1, 0] * jy
        phi_y = site.phase_tensors[-6].phi[0, 1] * jx + site.phase_tensors[-6].phi[0, 0] * jy
        # radii = np.sqrt(phi_x ** 2 + phi_y ** 2)
        # phi_min, phi_max = [np.min(radii), np.max(radii)]
        # phi_min, phi_max = [phi_min / phi_max, 1]
        ells.append([site.locations['Y'] / 1000 - phi_x / site.phase_tensors[-6].phi_max,
                     site.locations['X'] / 1000 - phi_y / site.phase_tensors[-6].phi_max])
        # ells.append(Ellipse(xy=(site.locations['Y'], site.locations['X']),
        #                     width=phi_max * 1000,
        #                     height=phi_min * 1000,
        #                     angle=90 - np.rad2deg(site.phase_tensors[-6].azimuth)))
    # print([phi_min, phi_max])
    # print(site.phase_tensors[-6].azimuth)
    fig = Figure()
    ax = fig.add_subplot(111, aspect='equal')
    vals = np.array([getattr(data.sites[site].phase_tensors[-6], fill_param) for site in data.site_names])
    vals = np.rad2deg(np.arctan(vals))
    norm_vals = utils.normalize(vals, lower=0, upper=90, explicit_bounds=True)
    for ii, e in enumerate(ells):
        # ax.add_artist(e)
        ax.fill(e[0], e[1], color=cmap(norm_vals[ii]))
        ax.annotate(data.site_names[ii][-3:], xy=(e[0][0], e[1][0]))
        # e.set_facecolor(cmap(norm_vals[ii]))
        # e.set_clip_box(ax.bbox)
    fake_im = ax.imshow(np.tile(np.arange(90), [2, 1]), cmap=cmap)
    fake_im.set_visible(False)
    fake_im.colorbar()
    ax.set_xlim(min(data.locations[:, 1] / 1000) - 5,
                max(data.locations[:, 1] / 1000) + 5)
    ax.set_ylim(min(data.locations[:, 0] / 1000) - 5,
                max(data.locations[:, 0] / 1000) + 5)
    ax.set_aspect('equal')

    # cb = ax.colorbar(cmap)
    # ax.set_xlim(-60000000, 10000000)
    # ax.set_ylim(-10000000, 10000000)
    return ells, vals, norm_vals


if __name__ == '__main__':
    # local_path = 'C:/Users/eroots'
    local_path = 'E:/phd/Nextcloud/'
    # filename = local_path + 'data/Regions/MetalEarth/AG/AG_plotset.dat'
    # listfile = local_path + 'data/Regions/MetalEarth/j2/upper_abitibi_hex.lst'
    # out_path = local_path + 'Documents/ME_transects/Upper_Abitibi/Paper/RoughFigures/PT/phi2_betaBack/betaCircle/'
    filename = local_path + 'data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wst_cullmantle3_LAMBERT_ZK_removed.dat'
    listfile = local_path + 'data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst'
    out_path = 'E:/phd/NextCloud/Documents/ME_Transects/wst/PTs/by_period/pt_only/phi2/'
    # filename = local_path + 'data/Regions/snorcle/j2/2020-collation-ian/grid_north.lst'
    # listfile = local_path + 'data/Regions/snorcle/j2/2020-collation-ian/grid_north.lst'
    # out_path = local_path + 'Documents/ME_transects/Upper_Abitibi/Paper/RoughFigures/PT/phi2_betaBack/betaCircle/'
    # jpg_file_name = local_path + 'ArcMap/AG/cio_georeferenced.jpg'
    # jpg_file_name = 'E:/phd/NextCloud/data/ArcMap/WST/WSBoundaries_Lambert_wMCR.jpg'
    jpg_file_name = ''
    out_file = 'wst_PT-phi2_'
    ext = ['.png', '.svg']
    dpi = 150
    padding = 20
    save_fig = 1
    bostick_depth = None
    cutoff_distance = 3500
    remove_close_sites = 0
    # fill_param = ['phi_2', 'beta']
    # fill_param = ['phi_split_pt', None]
    fill_param = ['phi_2', None]
    data = WSDS.Data(filename, listfile=listfile)
    raw = WSDS.RawData(listfile)
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
    raw.locations = raw.get_locs(mode='lambert')
    data.locations = raw.locations
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
    MV.colourmap = 'turbo'
    MV.phase_cax = [30, 90]
    MV.skew_cax = [-15, 15]
    MV.diff_cax = [-40, 40]
    # MV.interpolant = 'cubic'
    # MV.colourmap = 'bwr'
    # MV.colourmap = 'greys_r'
    MV.site_data['data'] = data
    MV.site_data['raw_data'] = raw
    MV.site_names = data.site_names
    MV.padding_scale = 10
    MV.pt_scale = .75
    MV.min_pt_ratio = 0.3
    MV.pt_ratio_cutoff = 0.01
    MV.phase_error_tol = 1000
    MV.rho_error_tol = 1000
    MV.include_outliers = True
    MV.allowed_std = 20
    MV.lut = 16
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
        # MV.plot_phase_tensor(data_type='data', normalize=True,
        #                      fill_param=fill_param[0], period_idx=ii)
        if (fill_param[0] != fill_param[1]) and fill_param[1]:
            two_param = 1
            # MV.use_colourbar = False
            # MV.colourmap = 'Reds'
            # MV.lut = 7
            # MV.plan_pseudosection(data_type='data', fill_param='beta',
                                  # n_interp=200, period_idx=ii)
            MV.use_colourbar = True
            # MV.lut = 32
            MV.colourmap = 'bwr'
            MV.min_pt_ratio = 1
            MV.pt_scale = 1.5
            MV.ellipse_linewidth = 0
            MV.plot_phase_tensor(data_type='data', normalize=True,
                             fill_param=fill_param[1], period_idx=ii,
                             bostick_depth=bostick_depth)
            # MV.colourmap = 'turbo'
            # MV.ellipse_linewidth = 1
            # MV.min_pt_ratio = 0.3
            # MV.pt_scale = 1.
            MV.plot_phase_tensor(data_type='data', normalize=True,
                             fill_param=fill_param[0], period_idx=ii,
                             bostick_depth=bostick_depth)
            # MV.plot_phase_bar(data_type='data', normalize=True,
                              # fill_param='beta', period_idx=ii)
        else:
            two_param = 0
            MV.plot_phase_tensor(data_type='data', normalize=True,
                             fill_param=fill_param[0], period_idx=ii,
                             bostick_depth=bostick_depth)
        # MV.plot_phase_bar2(data_type='data', normalize=True,
        #                    fill_param='phi_min', period_idx=ii)
        MV.set_axis_limits(bounds=[min(data.locations[:, 1]) - padding,
                                   max(data.locations[:, 1]) + padding,
                                   min(data.locations[:, 0]) - padding,
                                   max(data.locations[:, 0]) + padding])
        MV.window['axes'][0].set_aspect(1)
        MV.window['axes'][0].set_xlabel('Easting (m)', fontsize=14)
        MV.window['axes'][0].set_ylabel('Northing (m)', fontsize=14)
        if not two_param:
            label = MV.get_label(fill_param[0])
            MV.window['colorbar'].set_label(label + r' ($^{\circ}$)',
                                            rotation=270,
                                            labelpad=20,
                                            fontsize=18)
            caxes = [MV.window['colorbar'].ax]
        else:
            cax1 = MV.window['colorbar'].ax
            pos = MV.window['colorbar'].ax.get_position()
            cax1.set_aspect('auto')
            cax2 = MV.window['colorbar'].ax.twinx()
            # MV.window['colorbar'].ax.yaxis.set_label_position('left')
            if fill_param[1].lower() == 'beta':
                cax2.set_ylim([-10, 10])
                newlabel = [str(x) for x in range(-10, 12, 2)]
                cax2.set_yticks(range(-10, 12, 2))
                cax2.set_yticklabels(newlabel)
            elif fill_param[1].lower() == 'absbeta':
                cax2.set_ylim([0, 10])
                newlabel = [str(x) for x in range(0, 11, 1)]
                cax2.set_yticks(range(0, 11, 1))
                cax2.set_yticklabels(newlabel)
            pos.x0 += 0.05
            pos.x1 += 0.05
            cax1.set_position(pos)
            cax2.set_position(pos)
            label = MV.get_label(fill_param[0])
            cax1.set_ylabel(label + r' ($^{\circ}$)', fontsize=18)
            cax1.yaxis.set_ticks_position('right')
            cax1.yaxis.set_label_position('right')
            cax2.yaxis.set_ticks_position('left')
            cax2.yaxis.set_label_position('left')
            label = MV.get_label(fill_param[1])
            cax2.set_ylabel(label + r' ($^{\circ}$)', fontsize=18)
            caxes = [cax1, cax2]
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
