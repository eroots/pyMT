import numpy as np
import pyMT.data_structures as WSDS
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from e_colours import colourmaps as cm
import pyMT.utils as utils
from pyMT.gplot import MapView


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
    filename = 'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/GJH/ForEric/TNG&MTR-EDI/all.lst'
    data = WSDS.RawData(filename)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    MV = MapView(fig=fig)
    MV.site_data['raw_data'] = data
    MV.site_names = data.site_names
    # # MV.site_locations['generic'] = MV.get_locations(sites=MV.generic_sites)
    # MV.site_locations['active'] = MV.get_locations(
    #     sites=MV.active_sites)
    MV.site_locations['all'] = data.locations
    MV.plot_phase_tensor(data_type='raw_data', normalize=True,
                         fill_param='azimuth', period_idx=-6)
    # ells, vals, norm_vals = plot_ellipse(data, fill_param='phi_max')
    plt.show()
    