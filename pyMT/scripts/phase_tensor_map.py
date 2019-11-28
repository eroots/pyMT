import numpy as np
import pyMT.data_structures as WSDS
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from e_colours import colourmaps as cm
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
    # filename = 'F:/GJH/TNG&MTR-EDI/all.lst'
    # filename = 'C:/users/eroots/phd/ownCloud/data/ArcMap/LegacyMT/ag_edi/ag/all.lst'
    # filename = 'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/GJH/ForEric/TNG&MTR-EDI/all.lst'
    # filename = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/j2/cull_allSuperior.data'
    # listfile = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/j2/culled_allSuperior.lst'
    # out_path = 'C:/Users/eric/phd/ownCloud/Documents/Seminars/Seminar 3/Figures/PTs/'
    # filename = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/dry5/dry5_3.dat'
    # listfile = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/j2/dry5_3.lst'
    # out_path = 'C:/Users/eroots/phd/ownCloud/Documents/Dryden_paper/RoughFigures/PTs/'
    filename = 'C:/Users/eroots/phd/ownCloud/data/Regions/afton/sorted_lines.dat'
    listfile = 'C:/Users/eroots/phd/ownCloud/data/Regions/afton/j2/sorted_lines.lst'
    out_path = 'C:/Users/eroots/phd/ownCloud/Documents/TGI/Figures/'
    out_file = 'afton_PT_'
    ext = '.png'
    dpi = 600
    save_fig = 1
    cutoff_distance = 1000
    data = WSDS.Data(filename, listfile=listfile)
    raw = WSDS.RawData(listfile)
    # data.locations = rawdata.get_locs(mode='latlong')

    all_sites = deepcopy(data.site_names)
    # Remove redunantly close points
    # for ii, site1 in enumerate(data.site_names):
    #     for jj, site2 in enumerate(data.site_names):
    #         dist = euclidean((data.locations[ii, 1], data.locations[ii, 0]),
    #                          (data.locations[jj, 1], data.locations[jj, 0]))
    #         if dist < cutoff_distance and site1 in all_sites and (site1 != site2):
    #             if site2 in all_sites:
    #                 all_sites.remove(site2)
    # rm_sites = [site for site in data.site_names if site not in all_sites]
    # data.remove_sites(sites=rm_sites)
    # raw.remove_sites(sites=rm_sites)
    raw.locations = raw.get_locs(mode='latlong')
    for ii in range(len(raw.locations)):
        lon, lat = utils.project((raw.locations[ii, 1], raw.locations[ii, 0]), zone=10, letter='U')[2:]
        raw.locations[ii, 1], raw.locations[ii, 0] = lon, lat
    data.locations = raw.locations
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    MV = gplot.MapView(fig=fig)
    MV.window['figure'] = fig
    MV.window['axes'] = [ax]
    # MV.colourmap = 'jet'
    MV.colourmap = 'bwr'
    MV.site_data['data'] = data
    MV.site_names = data.site_names
    MV.padding_scale = 10
    MV.pt_scale = 1
    MV.phase_error_tol = 1000
    MV.rho_error_tol = 1000
    # # MV.site_locations['generic'] = MV.get_locations(sites=MV.generic_sites)
    # MV.site_locations['active'] = MV.get_locations(
    #     sites=MV.active_sites)
    MV.site_locations['all'] = data.locations
    # for ii in range(len(data.periods)):
    for ii in [30]:
        period = data.periods[ii]
        if period < 1:
            period = -1 / period
        period = str(int(period))
        MV.plot_phase_tensor(data_type='data', normalize=True,
                             fill_param='beta', period_idx=ii)
        MV.set_axis_limits(bounds=[min(data.locations[:, 1]) - 250,
                                   max(data.locations[:, 1]) + 250,
                                   min(data.locations[:, 0]) - 250,
                                   max(data.locations[:, 0]) + 250])
        MV.window['axes'][0].set_aspect(1)
        # ells, vals, norm_vals = plot_ellipse(data, fill_param='phi_max')
        if save_fig:
            plt.savefig(out_path + out_file + 'idx' + str(ii) + '_p' + period + ext, dpi=dpi,
                        transparent=True)
            ax.clear()
        else:
            plt.show()
