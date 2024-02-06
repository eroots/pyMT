import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure
from copy import deepcopy
from scipy.spatial.distance import euclidean
import shapefile
import numpy as np
import pyMT.data_structures as DS
import pyMT.utils as utils
from pyMT.e_colours import colourmaps as cm
import pyMT.gplot as gplot
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
import naturalneighbor as nn


def plot_shapefile(fig, shp_file_base,zorder=-1, units='km'):
    shp = shapereader.Reader(shp_file_base)
    # transform = ccrs.PlateCarree()
    # We want the data plotted in UTM, and we will convert them to UTM before plotting
    transform = ccrs.UTM(zone=17)
    transform.proj4_params['units'] = units
    # transform = ccrs.TransverseMercator(central_longitude=-85, central_latitude=49,
                                        # false_northing=5430000, false_easting=645000)
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111)
    ax = plt.axes(projection=transform)
    # ax.coastlines()
    for record, shape in zip(shp.records(), shp.geometries()):
        ax.add_geometries([shape], transform,
                          facecolor='none',
                          edgecolor='black',
                          zorder=zorder)
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
    # plt.grid(True)
    return fig, ax


if __name__ == '__main__':
    local_path = 'E:/'
    filename = local_path + 'phd/Nextcloud/data/Regions/MetalEarth/AG/AG_plotset.dat'
    model_file = local_path + 'phd/Nextcloud/data/Regions/MetalEarth/AG/Hex2Mod/HexAG_Z_static_rewrite.rho'
    listfile = local_path + 'phd/Nextcloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst'
    shp_file_base = local_path + 'phd/Nextcloud/data/ArcMap/Superior/MajorFaults_Superior.shp'
    # out_path = 'C:/Users/eric/phd/ownCloud/Documents/Seminars/Seminar 3/Figures/PTs/all/'
    # out_file = 'allSuperior_PT_'
    out_path = local_path + 'phd/Nextcloud/Documents/ME_Transects/Upper_Abitibi/Paper/RoughFigures/plan-views/wmap/bgy/'
    out_file = 'AG_planSlice_'
    ext = '.png'
    dpi = 600
    save_fig = 1
    cutoff_distance = 100
    plot_pts = 0
    remove_close_pts = 0
    plot_plan_view = 1
    plot_shape_file = 1
    slices = [5, 13, 16, 20, 22, 24, 26, 28, 30, 32, 33, 35, 37]
    # slices = [5]
    plot_stations = 1

#################################################################################
# READ FILES
    all_data = DS.Data(filename, listfile=listfile)
    all_raw = DS.RawData(listfile)
    model = DS.Model(model_file)
    # ME_data.remove_sites(sites=[site for site in ME_data.site_names if site.startswith('9')])
    # all_raw.remove_sites(sites=[site for site in all_raw.site_names if site.startswith('9')])
    # utm_number, utm_letter = 17, 'N'
    # all_raw.to_utm(zone=utm_number, letter=utm_letter)
    # all_data.locations = all_raw.locations
    # model.origin = all_raw.origin
    # model.to_UTM()
    model.spatial_units = 'm'
    all_data.spatial_units = 'm'
    all_raw.spatial_units = 'm'
    for z_slice in slices:
        fig = plt.figure(figsize=(16, 10))
        if plot_shape_file:
            fig, ax = plot_shapefile(fig=fig, shp_file_base=shp_file_base, zorder=3, units=all_data.spatial_units)
        else:
            ax = fig.add_subplot(111)
        MV = gplot.MapView(fig=fig)
        MV.window['figure'] = fig
        MV.window['axes'] = [ax]
        MV.colourmap = 'bgy_r'
        MV.model_cax = [0, 4.5]
        MV.lut = 64
        # MV.units = 'm'
        MV.site_data['data'], MV.site_data['raw_data'] = all_data, all_raw
        MV.model = model
        MV.site_names = all_data.site_names
        MV.padding_scale = 10
        # MV.pt_scale = 0.5
        MV.pt_scale = 1
        MV.phase_error_tol = np.inf
        MV.rho_error_tol = np.inf
        MV.use_colourbar = False
        MV.label_fontsize = 18
        MV.site_marker = 'v'
        MV.site_interior = 'w'
        MV.site_exterior['generic'] = 'k'
        MV.site_exterior['active'] = 'k'
        MV.edgewidth = 1
        MV.markersize = 8
        tick_fontsize = 14
        MV.site_locations['all'] = all_data.locations
        MV.set_locations()
        MV.coordinate_system = 'utm'
    #################################################################################
    # Plot Phase Tensors
        if plot_pts:
            # Remove redunantly close points
            if remove_close_pts:
                for ii, site1 in enumerate(all_data.site_names):
                    for jj, site2 in enumerate(all_data.site_names):
                        dist = euclidean((all_data.locations[ii, 1], all_data.locations[ii, 0]),
                                         (all_data.locations[jj, 1], all_data.locations[jj, 0]))
                        if dist < cutoff_distance and site1 in all_raw.site_names and (site1 != site2):
                            if site2 in all_raw.site_names:
                                all_raw.site_names.remove(site2)
                rm_sites = [site for site in all_data.site_names if site not in all_raw.site_names]
                all_data.remove_sites(sites=rm_sites)
                all_raw.remove_sites(sites=rm_sites)
                all_raw.locations = all_raw.get_locs(mode='latlong')
                for ii in range(len(all_raw.locations)):
                    lon, lat = utils.project((all_raw.locations[ii, 1],
                                              all_raw.locations[ii, 0]),
                                             zone=16, letter='U')[2:]
                    all_raw.locations[ii, 1], all_raw.locations[ii, 0] = lon, lat
                all_data.locations = all_raw.locations

                all_sites = deepcopy(all_data.site_names)
                # Remove redunantly close points
                for ii, site1 in enumerate(all_data.site_names):
                    for jj, site2 in enumerate(all_data.site_names):
                        dist = euclidean((all_data.locations[ii, 1], all_data.locations[ii, 0]),
                                         (all_data.locations[jj, 1], all_data.locations[jj, 0]))
                        if dist < cutoff_distance and site1 in all_sites and (site1 != site2):
                            if site2 in all_sites and not (site2 in all_raw.site_names):
                                all_sites.remove(site2)
                rm_sites = [site for site in all_data.site_names if site not in all_sites]
                all_data.remove_sites(sites=rm_sites)
                all_raw.remove_sites(sites=rm_sites)
                all_raw.locations = all_raw.get_locs(mode='latlong')
                for ii in range(len(all_raw.locations)):
                    lon, lat = utils.project((all_raw.locations[ii, 1],
                                              all_raw.locations[ii, 0]),
                                             zone=utm_number, letter=utm_letter)[2:]
                    all_raw.locations[ii, 1], all_raw.locations[ii, 0] = lon, lat
                all_data.locations = all_raw.locations
            ###############################################################################
                for ii, period in enumerate(all_data.periods[0:1]):
                    if period < 4:
                        use_data = deepcopy(all_data)
                    else:
                        use_data = deepcopy(all_data)
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

        if plot_plan_view:
            MV.plot_plan_view(z_slice=z_slice)

        if plot_stations:
            MV.plot_locations()

        MV.set_axis_limits()
        def kilometers(x, pos):
            return '{:4.0f}'.format(x / 1000)
        formatter = FuncFormatter(kilometers)
        MV.window['axes'][0].xaxis.set_major_formatter(formatter)
        MV.window['axes'][0].yaxis.set_major_formatter(formatter)
        for tick in MV.window['axes'][0].xaxis.get_major_ticks():
            tick.label.set_fontsize(tick_fontsize)
        for tick in MV.window['axes'][0].yaxis.get_major_ticks():
            tick.label.set_fontsize(tick_fontsize)

        if save_fig:
            plt.savefig(out_path + out_file + '{:.0f}'.format(model.dz[z_slice]) + ext, dpi=dpi,
                        transparent=True, bbox_inches='tight')
            # plt.savefig(out_path + out_file + str(ii) + '.pdf', dpi=dpi,
            #             transparent=True, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
