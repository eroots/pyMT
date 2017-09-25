import pyMT.data_structures as WSDS
import pyMT.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

datafile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\wst\New\wst0_5\wst0Inv_Final.data'
listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\wst\New\j2\cull6.lst'
respfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\wst\New\wst0_5\wst0Final_resp.00'
data = WSDS.Data(datafile=datafile, listfile=listfile)
raw = WSDS.RawData(listfile=listfile)
resp = WSDS.Data(datafile=respfile, listfile=listfile)
bdp = np.zeros((len(data.site_names), data.NP, 2))
bdp_resp = np.zeros((len(data.site_names), data.NP, 2))
narrow_periods = sorted(list(raw.narrow_periods.keys()))
for ii, site in enumerate(data.site_names):
    bostick, depth = utils.compute_bost1D(data.sites[site])[:2]
    bdp[ii, :, 0] = bostick
    bdp[ii, :, 1] = depth
    bostick, depth = utils.compute_bost1D(resp.sites[site])[:2]
    bdp_resp[ii, :, 0] = bostick
    bdp_resp[ii, :, 1] = depth

max_depth = np.max(bdp[:, :, 1], axis=0)
mean_depth_resp = np.mean(bdp_resp[:, :, 1], axis=0)
min_depth = np.min(bdp[:, :, 1], axis=0)
mean_depth = np.mean(bdp[:, :, 1], axis=0)

mean, = plt.loglog(data.periods, mean_depth, 'ro')
# med, = plt.loglog(data.periods, median_depth, 'bo')
mean_resp, = plt.loglog(data.periods, mean_depth_resp, 'bo')
mi, = plt.loglog(data.periods, min_depth, 'k--')
ma, = plt.loglog(data.periods, max_depth, 'k--')
# plt.legend([mean, med, mi, ma], ['Mean', 'Median', 'Minimum', 'Maximum'])
plt.legend([mean, mean_resp], ['Data Mean', 'Response Mean'], loc='upper left')
plt.gca().xaxis.set_ticks(np.logspace(0, 4.5, 10))
plt.gca().yaxis.set_ticks(np.logspace(0, 4, 9))
for ax in [plt.gca().xaxis, plt.gca().yaxis]:
    ax.set_major_formatter(ScalarFormatter())
plt.show()
