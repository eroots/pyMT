import numpy as np
import pyMT.data_structures as WSDS
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import e_colours.colourmaps as cm
import pyMT.utils as utils

cmap = cm.jet_plus()


def normalize_ellipse(phi):
    phi_min = abs(phi.phi_min)
    phi_max = abs(phi.phi_max)
    phi_min, phi_max = utils.normalize([phi_min, phi_max])
    return phi_min, phi_max


def plot_ellipse(data, fill_param):
    ells = []
    for site in data.sites.values():
        phi_min, phi_max = normalize_ellipse(site.phase_tensors[-1])
        ells.append(Ellipse(xy=(site.locations['Y'], site.locations['X']),
                            width=phi_max,
                            height=phi_min,
                            angle=site.phase_tensors[-1].azimuth))

    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    vals = np.array([getattr(data.sites[site].phase_tensors[-1], fill_param) for site in data.site_names])
    norm_vals = (vals - np.min(vals)) / \
                (np.max(vals) - np.min(vals))
    for ii, e in enumerate(ells):
        ax.add_artist(e)
        ax.set_facecolor(cmap(norm_vals[ii]))
        e.set_clip_box(ax.bbox)
    ax.set_xlim(min(data.locations[:, 0]), max(data.locations[:, 0]))
    ax.set_ylim(min(data.locations[:, 1]), max(data.locations[:, 1]))


if __name__ == '__main__':
    filename = 'F:/GJH/TNG&MTR-EDI/all.lst'
    data = WSDS.RawData(filename)
    plot_ellipse(data, fill_param='beta')
    plt.show()
    