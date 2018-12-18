from matplotlib.figure import Figure
import pyMT.gplot as gplot
from copy import deepcopy
from scipy.spatial.distance import euclidean
import shapefile
import numpy as np
import matplotlib.pyplot as plt
import pyMT.data_structures as WSDS
import pyMT.utils as utils
import cartopy.crs as ccrs
from cartopy.io import shapereader
import matplotlib
import naturalneighbor as nn
import e_colours.colourmaps as cm

cmap = cm.jet()


def plot_shapefile(shp_file_base):
    shp = shapereader.Reader(shp_file_base)
    # transform = ccrs.PlateCarree()
    # We want the data plotted in UTM, and we will convert them to UTM before plotting
    # transform = ccrs.UTM(zone=16)
    transform = ccrs.TransverseMercator(central_longitude=-85, central_latitude=49,
                                        false_northing=5430000, false_easting=645000)
    fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111)
    ax = plt.axes(projection=transform)
    ax.coastlines()
    for record, shape in zip(shp.records(), shp.geometries()):
        ax.add_geometries([shape], ccrs.PlateCarree(),
                          facecolor='lightgrey',
                          edgecolor='black',
                          zorder=-1)
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
    return fig


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
    filename = 'C:/Users/eric/phd/ownCloud/secondary/simpeg_tests/eastSuperior_1D_Z.dat'
    listfile = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/j2/1D_sites.lst'
    shp_file_base = 'C:/Users/eric/phd/ownCloud/data/ArcMap/test2.shp'
    # out_path = 'C:/Users/eric/phd/ownCloud/Documents/Seminars/Seminar 3/Figures/PTs/all/'
    # out_file = 'allSuperior_PT_'
    out_path = 'C:/Users/eric/phd/ownCloud/secondary/simpeg_tests/Figures/PTs/'
    out_file = '1D_PT_'
    ext = '.png'
    dpi = 600
    save_fig = 1
    cutoff_distance = 500

#################################################################################
    all_data = WSDS.Data(filename, listfile=listfile)
    all_raw = WSDS.RawData(listfile)
    ME_data = deepcopy(all_data)
    ME_raw = deepcopy(all_raw)
    ME_data.remove_sites(sites=[site for site in ME_data.site_names if site.startswith('9')])
    ME_raw.remove_sites(sites=[site for site in ME_raw.site_names if site.startswith('9')])
    all_sites_ME = deepcopy(ME_data.site_names)
    # Remove redunantly close points
    for ii, site1 in enumerate(ME_data.site_names):
        for jj, site2 in enumerate(ME_data.site_names):
            dist = euclidean((ME_data.locations[ii, 1], ME_data.locations[ii, 0]),
                             (ME_data.locations[jj, 1], ME_data.locations[jj, 0]))
            if dist < cutoff_distance and site1 in all_sites_ME and (site1 != site2):
                if site2 in all_sites_ME:
                    all_sites_ME.remove(site2)
    rm_sites = [site for site in ME_data.site_names if site not in all_sites_ME]
    ME_data.remove_sites(sites=rm_sites)
    ME_raw.remove_sites(sites=rm_sites)
    ME_raw.locations = ME_raw.get_locs(mode='latlong')
    for ii in range(len(ME_raw.locations)):
        lon, lat = utils.project((ME_raw.locations[ii, 1],
                                  ME_raw.locations[ii, 0]),
                                 zone=16, letter='U')[2:]
        ME_raw.locations[ii, 1], ME_raw.locations[ii, 0] = lon, lat
    ME_data.locations = ME_raw.locations

    all_sites = deepcopy(all_data.site_names)
    # Remove redunantly close points
    for ii, site1 in enumerate(all_data.site_names):
        for jj, site2 in enumerate(all_data.site_names):
            dist = euclidean((all_data.locations[ii, 1], all_data.locations[ii, 0]),
                             (all_data.locations[jj, 1], all_data.locations[jj, 0]))
            if dist < cutoff_distance and site1 in all_sites and (site1 != site2):
                if site2 in all_sites and not (site2 in all_sites_ME):
                    all_sites.remove(site2)
    rm_sites = [site for site in all_data.site_names if site not in all_sites]
    all_data.remove_sites(sites=rm_sites)
    all_raw.remove_sites(sites=rm_sites)
    all_raw.locations = all_raw.get_locs(mode='latlong')
    for ii in range(len(all_raw.locations)):
        lon, lat = utils.project((all_raw.locations[ii, 1],
                                  all_raw.locations[ii, 0]),
                                 zone=16, letter='U')[2:]
        all_raw.locations[ii, 1], all_raw.locations[ii, 0] = lon, lat
    all_data.locations = all_raw.locations
###############################################################################
    for ii, period in enumerate(all_data.periods):
        if period < 4:
            use_data = deepcopy(ME_data)
        else:
            use_data = deepcopy(all_data)
        fig = plot_shapefile(shp_file_base)
        MV = gplot.MapView(fig=fig)
        # MV.colourmap = 'jet'
        MV.colourmap = 'bwr'
        MV.site_data['data'] = use_data
        MV.site_names = use_data.site_names
        MV.padding_scale = 10
        # MV.pt_scale = 0.5
        MV.pt_scale = 1
        MV.phase_error_tol = np.inf
        MV.rho_error_tol = np.inf
        MV.use_colourbar = False
        MV.site_locations['all'] = use_data.locations
        MV.plot_phase_tensor(data_type='data', normalize=True,
                             fill_param='beta', period_idx=ii)
        MV.window['axes'][0].set_title('Frequency: {:5.5g} Hz, Period: {:5.5g} s'.format(1 / period, period), fontsize=18)
        MV.window['axes'][0].set_xlabel('Easting (m)', fontsize=18)
        MV.window['axes'][0].set_ylabel('Northing (m)', fontsize=18)
        for label in MV.window['axes'][0].xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        # ax.axes().set_aspect('equal')
        MV.window['axes'][0].tick_params(axis='both', labelsize=14)
        # MV.window['axes'][0].set_xlim([-100000, 1500000])
        MV.window['axes'][0].set_xlim([790000, 1200000])
        # cax, kw = matplotlib.colorbar.make_axes(MV.window['axes'],
        #                                         location='bottom',
        #                                         pad=0.125,
        #                                         shrink=0.9,
        #                                         extend='both')
        # cb = fig.colorbar(MV.fake_im, cax=cax, **kw)
        # cb.set_label(r'$\beta (^\circ$)',
        #              rotation=0,
        #              labelpad=10,
        #              fontsize=14)
        # MV.plot_locations()
        # ells, vals, norm_vals = plot_ellipse(data, fill_param='phi_max')
        if save_fig:
            plt.savefig(out_path + out_file + str(ii) + ext, dpi=dpi,
                        transparent=True, bbox_inches='tight')
            # plt.savefig(out_path + out_file + str(ii) + '.pdf', dpi=dpi,
            #             transparent=True, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
