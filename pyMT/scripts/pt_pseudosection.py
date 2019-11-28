import pyMT.data_structures as WSDS
import matplotlib.pyplot as plt
import numpy as np
from pyMT.e_colours import colourmaps as cm
import pyMT.utils as utils
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes


local_path = 'C:/Users/eroots/'


def generate_ellipse(phi):
    step_size = np.pi / 30
    jx = np.cos(np.arange(0, 2 * np.pi + step_size, step_size))
    jy = np.sin(np.arange(0, 2 * np.pi + step_size, step_size))
    phi_x = phi[0, 0] * jx + phi[0, 1] * jy
    phi_y = phi[1, 0] * jx + phi[1, 1] * jy
    return phi_x, phi_y


# listfile = r'C:\Users\eric\phd\Kilauea\ConvertedEDIs\2018-517\allsites.lst'
# listfile = r'C:\Users\eric\phd\Kilauea\ConvertedEDIs\all\allsites.lst'
# listfile = r'C:\Users\eric\phd\Kilauea\ConvertedEDIs\all\515-520.lst'
listfile = local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/north_main_transect_all.lst'
main_list = local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/north_main_transect_all.lst'
# listfile = 'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/MetalEarth/swayze/j2/southeast_R2.lst'
# main_list = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/southeast_R2.lst'
# listfile = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/swz_cull1.lst'
# listfile = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/main_transect_more.lst'
# main_list = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/main_transect_more.lst'
# listfile = local_path + 'phd/ownCloud/data/Regions/MetalEarth/matheson/j2/mat_eastLine.lst'
# main_list = local_path + 'phd/ownCloud/data/Regions/MetalEarth/matheson/j2/mat_eastLine.lst'
# main_list = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/larder/j2/main_transect_bb_NS.lst'
# listfile = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/larder/j2/main_transect_bb_NS.lst'
# datafile = 'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/MetalEarth/swayze/test.data'
# datafile = 'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/MetalEarth/swayze/swz_cull1/finish/swz_cull1i.data'
# datafile = 'F:/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/finish/swz_cull1i.data'
# datafile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\sturgeon\stu3\stu2_j2Rot2.dat')
# main_list = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/j2/main_transect.lst'
# listfile = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/j2/main_transect.lst'
###########################################
# DRYDEN
# listfile = local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/j2/dry5_3.lst'
# main_list = local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/j2/main_transect_pt.lst'
###########################################
# MALARTIC
# listfile = local_path + 'phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal_amt.lst'
# main_list = local_path + 'phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal_amt.lst'
###########################################
# LARDER
# listfile = local_path + 'phd/ownCloud/data/Regions/MetalEarth/larder/j2/main_transect_bb.lst'
# main_list = local_path + 'phd/ownCloud/data/Regions/MetalEarth/larder/j2/main_transect_bb.lst'
###########################################
# NEW AFTON
# for use_list in ['l0', 'l3', 'l6', 'l9', 'l12', 'l15', 'l18', 'l21']:
# for use_list in ['l3']:
for use_list_dummy in [0]:
    # listfile = local_path + 'phd/ownCloud/data/Regions/afton/j2/sorted_lines.lst'
    # main_list = local_path + 'phd/ownCloud/data/Regions/afton/j2/l9.lst'
    # main_list = local_path + 'phd/ownCloud/data/Regions/afton/j2/' + use_list + '.lst'

    data = WSDS.RawData(listfile)
    main_transect = WSDS.RawData(main_list)
    UTM_number = 17
    UTM_letter = 'N'

    # data = WSDS.RawData(listfile=listfile)
    normalize = 1
    fill_param = 'phi_2'
    # fill_param = 'beta'
    # fill_param = 'Lambda'
    # use_periods = sorted(list(data.narrow_periods.keys()))
    # all_periods = set(list(data.sites['SWZ047A'].periods) + list(data.sites['SWZ064M'].periods))
    all_periods = set(list(data.sites['18-swz003a'].periods) + list(data.sites['18-swz006m'].periods))
    use_periods = sorted([p for p in all_periods if p < 15])
    high_cut = 15
    # use_periods = sorted([p for p in data.narrow_period_list(count_tol=0.1, high_tol=0.1).keys() if p < 150])
    # use_periods = data.sites[data.site_names[0]].periods
    x_scale, y_scale = 1, 1.25
    save_fig = 0
    freq_skip = 1
    radius = 0.09
    # radius = 0.1
    # radius = 1
    label_offset = -4.5
    # label_offset = -3.7
    annotate_sites = 0
    use_colourbar = 0
    # file_path = local_path + '/phd/ownCloud/Documents/TGI/Figures/PT_pseudosections/'
    file_path = local_path + '/phd/ownCloud/Documents/ME_transects/Swayze_paper/RoughFigures/PT_round2/'
    file_name = 'pt_pseudosection_phi2_north_all_linear'
    # file_name = '{}_{}'.format(use_list, fill_param)
    file_types = ['.png', '.svg']
    dpi = 600
    linear_xaxis = 1
    # cmap = cm.jet_plus_r(64)
    # cmap = cm.get_cmap('turbo', 64)
    # cmap = cm.bwr(32)
    cmap = cm.get_cmap('turbo', 32)

    data.locations = data.get_locs(mode='latlong')
    main_transect.locations = main_transect.get_locs(mode='latlong')
    for ii, site in enumerate(data.site_names):
        easting, northing = utils.project((data.locations[ii, 1],
                                           data.locations[ii, 0]),
                                          zone=UTM_number, letter=UTM_letter)[2:]
        data.locations[ii, 1], data.locations[ii, 0] = easting, northing
        data.sites[site].locations['X'], data.sites[site].locations['Y'] = northing, easting
    for ii, site in enumerate(main_transect.site_names):
        easting, northing = utils.project((main_transect.locations[ii, 1],
                                           main_transect.locations[ii, 0]),
                                          zone=UTM_number, letter=UTM_letter)[2:]
        main_transect.locations[ii, 1], main_transect.locations[ii, 0] = easting, northing
        main_transect.sites[site].locations['X'], main_transect.sites[site].locations['Y'] = northing, easting
    main_transect.spatial_units = 'km'
    data.spatial_units = 'km'
    if linear_xaxis:
        # Make sure the sites go north-south
        main_transect.locations = main_transect.locations[main_transect.locations[:, 0].argsort()]
        # Sort the site names so the same is true
        main_transect.site_names = sorted(main_transect.site_names,
                                      key=lambda x: main_transect.sites[x].locations['X'])
        linear_x = np.zeros(main_transect.locations.shape[0])
        linear_x[1:] = np.sqrt((main_transect.locations[1:, 0] - main_transect.locations[:-1, 0]) ** 2 +
                               (main_transect.locations[1:, 1] - main_transect.locations[:-1, 1]) ** 2)
        linear_x = np.cumsum(linear_x)
        nodes = np.array([main_transect.locations[:, 0], main_transect.locations[:, 1]]).T
        linear_site = np.zeros((len(main_transect.locations)))
        for ii, (x, y) in enumerate(main_transect.locations):
            dist = np.sum((nodes - np.array([x, y])) ** 2, axis=1)
            idx = np.argmin(dist)
            linear_site[ii] = linear_x[idx]
        x_lim = [0, linear_site[-1]]
        # x_scale = (linear_site[-1] - linear_site[0])
    else:
        x_lim = [min(main_transect.locations[:, 0]) - 0.1,
                 max(main_transect.locations[:, 0]) + 0.1]
    # scale = np.sqrt((len(data.site_names) - 1) ** 2 +
    #                 (np.log10(np.max(periods)) -
    #                  np.log10(np.min(periods))) ** 2)
    #     x_scale = np.sqrt((np.max(data.locations[:, 0] / 1000) -
    #                        np.min(data.locations[:, 0]) / 1000) ** 2)
    # y_scale = np.sqrt((np.log10(np.max(use_periods)) -
                       # np.log10(np.min(use_periods))) ** 2)

    # scale = 2
    periods = []
    loc = []
    X = []
    Y = []
    fill_vals = []
    ellipses = []
    main_sites = []
    good_periods, questionable_periods, not_perfectly_matched_but_still_ok_periods = 0, 0, 0
    for ii, site_name in enumerate(main_transect.site_names):
        if site_name in main_transect.site_names:
            main_sites.append(site_name)
            site = data.sites[site_name]
            for jj, period in enumerate(site.periods):
            # for jj, period in enumerate(use_periods):
                # if jj % 2 == 0 and period < 1100 and period >= 1 / 300:
                if (jj % (freq_skip + 1) == 0) and (period < high_cut):
                    kk = np.argmin(abs(period - site.periods))
                    if period != site.periods[kk]:
                        if 100 * abs(period - site.periods[kk]) / period > 10:
                            questionable_periods += 1
                            # continue
                        else:
                            not_perfectly_matched_but_still_ok_periods += 1
                    else:
                        good_periods += 1
                        # print('Period used does not match period of site {}'.format(site.name))
                        # print('Discrepancy of {}%'.format(100 * (period - site.periods[kk]) / period))
                    periods.append(np.log10(period))
                    # loc_x = ii
                    # loc_x = site.locations['Lat'] # / 1000
                    if linear_xaxis:
                        loc_x = linear_site[ii]
                        # loc_x = ii
                    else:
                        loc_x = site.locations['X']
                    loc.append(loc_x)
                    phi_x, phi_y = generate_ellipse(site.phase_tensors[kk].phi)
                    #  These numbers may have to be changed to scale with the width of the plot
                    phi_x, phi_y = (phi_x / np.abs(site.phase_tensors[kk].phi_max),
                                    phi_y / np.abs(site.phase_tensors[kk].phi_max))
                    # radius = np.max(np.sqrt(phi_x ** 2 + phi_y ** 2))
                    # radius = np.max(np.maximum(phi_x, phi_y))
                    # radius = 100
                    # if radius > 1000:
                    # phi_x, phi_y = [(scale / (radius * 1000)) * x for x in (phi_x, phi_y)]
                    phi_x *= (radius / (x_scale))
                    phi_y *= (radius / (y_scale))
                    y_max = np.max(phi_y) - np.min(phi_y)
                    x_max = np.max(phi_x) - np.min(phi_x)
                    if x_max * 5 < y_max:
                        phi_x *= 10
                    elif y_max * 5 < x_max:
                        phi_y *= 10
                    # ellipses.append([np.log10(period) - phi_x, ii - phi_y])
                    ellipses.append([loc_x - phi_x, np.log10(period) - phi_y])
                    fill_vals.append(getattr(site.phase_tensors[kk], fill_param))
    # xticks = np.arange(0, loc[-1], 24)
    # xtick_labels = [str(x) for x in np.arange(501, 531)]
    print('Number of mismatched periods with >10% difference: {}'.format(questionable_periods))
    print('Number of mismatched periods with <10% difference: {}'.format(not_perfectly_matched_but_still_ok_periods))
    print('Number of perfectly matched periods: {}'.format(good_periods))
    fill_vals = np.array(fill_vals)
    if fill_param in ['phi_max', 'phi_min', 'det_phi', ' phi_1', 'phi_2', 'phi_3']:
        lower, upper = (0, 90)
        # cmap = cm.jet_plus_r(64)
        fill_vals = np.rad2deg(np.arctan(fill_vals))
    elif fill_param in ['Lambda']:
        lower, upper = (0, 1)
        # cmap = cm.jet_plus_r(64)
    elif fill_param == 'beta':
        lower, upper = (-10, 10)
        # cmap = cm.bwr(64)
        fill_vals = np.rad2deg(np.arctan(fill_vals))
    elif fill_param in ['alpha', 'azimuth']:
        lower, upper = (-90, 90)
        # cmap = cm.bwr(64)
        fill_vals = np.rad2deg(np.arctan(fill_vals))
    fill_vals[fill_vals > upper] = upper
    fill_vals[fill_vals < lower] = lower
    norm_vals = utils.normalize_range(fill_vals,
                                      lower_range=lower,
                                      upper_range=upper,
                                      lower_norm=0,
                                      upper_norm=1)


    def plot_it():
        # fig = plt.figure(1, figsize=(8, 4.5))
        fig = plt.figure(1, figsize=(16, 12))
        # h = [Size.Fixed(0.), Size.Fixed(6.5)]
        # v = [Size.Fixed(0.5), Size.Fixed(3.25)]
        h = [Size.Fixed(0.), Size.Fixed(13)]
        v = [Size.Fixed(0.5), Size.Fixed(7)]
        win = Divider(fig, (0.1, 0.08, 0.8, 0.8), h, v, aspect=False)
        ax = Axes(fig, win.get_position())
        ax.set_axes_locator(win.new_locator(nx=1, ny=1))
        fig.add_axes(ax)
        # fig = plt.figure(figsize=(16, 12))
        # ax = fig.add_subplot(111)
        for ii, ellipse in enumerate(ellipses):
            ax.fill(ellipse[0], ellipse[1],
                    color=cmap(norm_vals[ii]),
                    zorder=0)
            ax.plot(ellipse[0], ellipse[1],
                    'k-', linewidth=0.2)
        ax.invert_yaxis()
        plt.xlabel('Northing (km)', fontsize=16)
        plt.ylabel(r'$\log_{10}$ Period (s)', fontsize=16)
        # ax.set_aspect(1)
        ax.tick_params(axis='both', labelsize=14)
        # locs, labels = plt.xticks()
        # plt.xticks(locs, [int(x * 10) for x in locs])
        fake_vals = np.linspace(lower, upper, len(fill_vals))
        fake_im = ax.scatter(loc,
                             periods,
                             c=fake_vals, cmap=cmap)
        fake_im.set_visible(False)
        # ax.set_ylim([-2.6, 3.25])
        # ax.set_xlim([526.5, 538.2])
        # ax.invert_yaxis()
        # cb = plt.colorbar(mappable=fake_im)
        #############################################
        # Colour bar and site labelling
        if use_colourbar:
            # cbaxes = fig.add_axes([0.925, 0.1351, 0.015, 0.72])
            cb = plt.colorbar(fake_im)
            if 'phi' in fill_param[:3]:
                label = r'${}{}(\degree)$'.format('\phi', fill_param[-2:])
            else:
                label = r'$\{}(\degree)$'.format(fill_param)
            cb.set_label(label,
                         rotation=270,
                         labelpad=20,
                         fontsize=18)
        # ax.tick_params(axis='both', labelsize=14)
        # ax.set_xlim(x_lim)
        if annotate_sites:
            for ii, site in enumerate(main_sites):
                txt = site[-4:-1]
                if linear_xaxis:
                    ax.text(linear_site[ii],
                            label_offset, site, rotation=45)  # 3.6
                else:
                    ax.text(main_transect.sites[site].locations['X'],
                            label_offset, site, rotation=45)  # 3.6
        return fig

    fig = plot_it()
    if save_fig:
        for ext in file_types:
            fig.savefig(file_path + file_name + ext, dpi=dpi,
                        transparent=True)
        plt.close()
    else:
        plt.show()
        # plt.plot([main_transect.sites[site].locations['X'] / 10000,
        #           main_transect.sites[site].locations['X'] / 10000],
                 # [-2.75, -2.7], 'k-')
        # plt.text(main_transect.sites[site].locations['X'] / 10000 - 0.05, -2.95, txt, rotation=50)

    # plt.show()
    # plt.savefig('F:/ownCloud/Documents/Swayze_paper/Figures/pt_pseudosection_beta.ps',
    #             dpi=600, orientation='landscape')
    # fig, ax = plt.subplots(figsize=(16, 12))
    # plt.colorbar(fake_im, ax=ax)
    # ax.remove()
    # plt.savefig('F:/ownCloud/Documents/Swayze_paper/Figures/phase_colourbar_beta.ps', dpi=600)

# def plot_phase_tensor(data, normalize=True, fill_param='Beta'):
# def generate_ellipse(phi):
#     jx = np.cos(np.arange(0, 2 * np.pi, np.pi / 30))
#     jy = np.sin(np.arange(0, 2 * np.pi, np.pi / 30))
#     phi_x = phi[0, 0] * jx + phi[0, 1] * jy
#     phi_y = phi[1, 0] * jx + phi[1, 1] * jy
#     return phi_x, phi_y
# ellipses = []
# if fill_param != 'Lambda':
#     fill_param = fill_param.lower()
# # if fill_param == 'azimuth':
# #     cmap = cm.hsv()
# # else:
# cmap = cm.jet()
# X_all, Y_all = self.site_locations['all'][:, 0], self.site_locations['all'][:, 1]
# scale = np.sqrt((np.max(X_all) - np.min(X_all)) ** 2 +
#                 (np.max(Y_all) - np.min(Y_all)) ** 2)
# for ii, site_name in enumerate(self.site_names):
#     site = self.site_data[data_type].sites[site_name]
#     phi_x, phi_y = generate_ellipse(site.phase_tensors[period_idx].phi)
#     X, Y = X_all[ii], Y_all[ii]
#     phi_x, phi_y = (1000 * phi_x / site.phase_tensors[period_idx].phi_max,
#                     1000 * phi_y / site.phase_tensors[period_idx].phi_max)
#     radius = np.max(np.sqrt(phi_x ** 2 + phi_y ** 2))
#     # if radius > 1000:
#     phi_x, phi_y = [(5 * scale / (radius * 100)) * x for x in (phi_x, phi_y)]
#     ellipses.append([Y - phi_x, X - phi_y])
# fill_vals = np.array([getattr(self.site_data[data_type].sites[site].phase_tensors[period_idx],
#                               fill_param)
#                       for site in self.site_names])
# if fill_param in ['phi_max', 'phi_min', 'det_phi', ' phi_1', 'phi_2', 'phi_3']:
#     lower, upper = (0, 90)
# elif fill_param in ['Lambda']:
#     lower, upper = (np.min(fill_vals), np.max(fill_vals))
# elif fill_param == 'beta':
#     lower, upper = (-10, 10)
# elif fill_param in ['alpha', 'azimuth']:
#     lower, upper = (-90, 90)
# fill_vals = np.rad2deg(np.arctan(fill_vals))
# fill_vals[fill_vals > upper] = upper
# fill_vals[fill_vals < lower] = lower
# norm_vals = utils.normalize_range(fill_vals,
#                                   lower_range=lower,
#                                   upper_range=upper,
#                                   lower_norm=0,
#                                   upper_norm=1)
# for ii, ellipse in enumerate(ellipses):
#     self.window['axes'][0].fill(ellipse[0], ellipse[1],
#                                 color=cmap(norm_vals[ii]),
#                                 zorder=0)
# fake_vals = np.linspace(lower, upper, len(fill_vals))
# fake_im = self.window['axes'][0].scatter(self.site_locations['all'][:, 1],
#                                          self.site_locations['all'][:, 0],
#                                          c=fake_vals, cmap=cmap)
# fake_im.set_visible(False)
# cb = self.window['colorbar'] = self.window['figure'].colorbar(mappable=fake_im)
# cb.set_label(fill_param,
#              rotation=270,
#              labelpad=20,
#              fontsize=18)
