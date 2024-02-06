import pyMT.data_structures as WSDS
import pyMT.utils as utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.image import PcolorImage
import numpy as np
from pyMT.e_colours import colourmaps
import segyio


seismic_data_path = r'D:\OpenDTect_data\RawData\snorcle\feat_line3a_segy\line3_file22_curvelet.sgy'
clip_val = 0.5
seismic_is_depth = 0
depth_conversion_velocity = 6.3
aspect_ratio = 'auto'
zlim = [0, 50]
cmap = 'bwr'
with segyio.open(seismic_data_path, strict=False) as f:
    seis_data = np.stack([t.astype(np.float) for t in f.trace])
    x = np.array([t[segyio.TraceField.CDP_X] for t in f.header])
    y = np.array([t[segyio.TraceField.CDP_Y] for t in f.header])
    cdp = np.array([t[segyio.TraceField.CDP] for t in f.header])
    bad_idx = (x == 0) | (y == 0) | (cdp == 0)
    x = x[~bad_idx]
    y = y[~bad_idx]
    cdp = cdp[~bad_idx]
    # header = f.text[0].decode('ascii')
    dt = f.bin[segyio.BinField.Interval] / 1000
    samples = f.samples
seis_data = seis_data.T
# linear_x = np.zeros(x.shape)
# linear_x[1:] = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
# linear_x = np.cumsum(linear_x)
x_axis = cdp * 25 / 10
# if force_NS:
#     if plot_direction == 'sn':
#             if y[0] > y[-1]:
#                     seis_data = np.fliplr(seis_data)
#     elif plot_direction == 'we':
#             if x[0] > x[-1]:
#                     seis_data = np.fliplr(seis_data)
#     elif plot_direction == 'ns':
#             if y[0] < y[-1]:
#                     seis_data = np.fliplr(seis_data)
#     elif plot_direction == 'ew':
#             if x[0] < x[-1]:
#                     seis_data = np.fliplr(seis_data)
# seis_data = seis_data / np.linalg.norm(seis_data, axis=0)
# seis_data[:250,:] = 0
# if normalize_seismic:
seis_data = seis_data / np.linalg.norm(seis_data, axis=0)
# seis_data = np.fliplr(seis_data)
clip = clip_val*np.max(np.abs(seis_data))
seis_data[seis_data<-clip] = -clip
seis_data[seis_data>clip] = clip
# seis_data = np.abs(seis_data)
# alpha = (seis_data) / (np.max(seis_data) * 0.9)
alpha = 1
if seismic_is_depth:
    dvec = np.array(samples) / 1000
else:
    tvec = np.arange(seis_data.shape[0]) * dt / 1000
    dvec = tvec * depth_conversion_velocity / 2
plt.imshow((seis_data), 
                cmap=cmap,
                extent=[x_axis[0], x_axis[-1], min(dvec[-1], zlim[1]), dvec[0]],
                alpha=alpha, interpolation='sinc')
plt.gca().set_aspect(aspect_ratio)
plt.show()