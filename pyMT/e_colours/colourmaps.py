from scipy.interpolate import interp2d
from matplotlib import colors, cm
import numpy as np
from e_colours.utils import rgb2hls, hls2rgb
import colorcet  # This needs to be imported to register the cet colourmaps into mpl


COLOUR_MAPS = ('jet', 'jet_r', 'jet_plus', 'jet_plus_r',
               'bwr', 'bwr_r', 'bgy', 'bgy_r')


def jet_plus(N=64):
    cmap = np.array(([1, 0, 0],
                     [0.2, 1, 0],
                     [0, 0.2, 1]))
    [X, Y] = np.meshgrid(np.arange(3), np.arange(N))
    idx = [0, round(N / 2) - 1, N - 1]
    f = interp2d(np.arange(3), idx, cmap)
    cmap = f(np.arange(3), np.arange(N))
    hls = rgb2hls(cmap)
    hls[:, 1] = np.linspace(0.5, 0.4, N)
    jet_plus = hls2rgb(hls)
    jet_plus = colors.ListedColormap(hls2rgb(hls), 'jet_plus')
    return jet_plus


def jet_plus_r(N=64):
    cmap = np.array(([1, 0, 0],
                     [0.2, 1, 0],
                     [0, 0.2, 1]))
    [X, Y] = np.meshgrid(np.arange(3), np.arange(N))
    idx = [0, round(N / 2) - 1, N - 1]
    f = interp2d(np.arange(3), idx, cmap)
    cmap = f(np.arange(3), np.arange(N))
    hls = rgb2hls(cmap)
    hls[:, 1] = np.linspace(0.5, 0.4, N)
    jet_plus = colors.ListedColormap(hls2rgb(np.flip(hls, 0)), 'jet_plus_r')
    return jet_plus


def jet(N=64):
    return cm.get_cmap('jet', lut=N)


def jet_r(N=64):
    return cm.get_cmap('jet_r', lut=N)


def hsv(N=64):
    return cm.get_cmap('hsv', lut=N)


def bgy(N=64):
    return cm.get_cmap('cet_bgy', lut=N)


def bgy_r(N=64):
    return cm.get_cmap('cet_bgy_r', lut=N)


def bwr(N=64):
    return cm.get_cmap('bwr', lut=N)


def bwr_r(N=64):
    return cm.get_cmap('bwr_r', lut=N)


def greys(N=64):
    return cm.get_cmap('gray', lut=N)


def greys_r(N=64):
    return cm.get_cmap('gray_r', lut=N)


def get_cmap(cmap, N=64):
    if cmap in ('cet_bgy', 'bgy'):
        return bgy(N)
    elif cmap in('cet_bgy_r', 'bgy_r'):
        return bgy_r(N)
    elif cmap in('jet'):
        return jet(N)
    elif cmap in ('jet_r'):
        return jet_r(N)
    elif cmap in ('jet_plus'):
        return jet_plus(N)
    elif cmap in ('jet_plus_r'):
        return jet_plus_r(N)
    elif cmap in ('bwr'):
        return bwr(N)
    elif cmap in ('bwr_r'):
        return bwr_r(N)
    elif cmap in ('greys'):
        return greys(N)
    elif cmap in ('greys_r'):
        return greys_r(N)
