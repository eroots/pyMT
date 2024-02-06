import pyMT.data_structures as DS
import numpy as np
import pyMT.utils as utils
import matplotlib.pyplot as plt


data = DS.RawData('E:/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst-only.lst')
data.locations = data.get_locs(mode='lambert')
p_interest = 106
rm, p_idx = [], []
for ii, s in enumerate(data.site_names):
	if ((data.locations[ii, 0] < 0) or (data.locations[ii,0] > 0.25e6)
		or (data.locations[ii, 1] > 400000)):
		rm.append(s)
	else:
		p_idx.append(np.argmin(abs(data.sites[s].periods - p_interest)))

data.remove_sites(sites=rm)
data.locations = data.get_locs(mode='lambert')
# phase_xy = [utils.compute_phase(data.sites[site], calc_comp='xy', wrap=1)[0][p_idx[ii]] for ii, site in enumerate(data.site_names)]
# phase_yx = [utils.compute_phase(data.sites[site], calc_comp='yx', wrap=1)[0][p_idx[ii]] for ii, site in enumerate(data.site_names)]
phase_xy = [np.rad2deg(np.arctan(data.sites[site].phase_tensors[p_idx[ii]].phi_max)) for ii, site in enumerate(data.site_names)]
phase_yx = [np.rad2deg(np.arctan(data.sites[site].phase_tensors[p_idx[ii]].phi_min)) for ii, site in enumerate(data.site_names)]
phase_diff = [phase_xy[ii] - phase_yx[ii] for ii in range(data.NS)]

fig, ax = plt.subplots(nrows=3, ncols=1)
for ii, to_plot in enumerate([phase_xy, phase_yx, phase_diff]):
	ax[ii].plot(to_plot, 'k-')
	ax[ii].plot(range(len(to_plot)), np.ones(len(to_plot)) * np.min(to_plot), 'k--')
	ax[ii].plot(range(len(to_plot)), np.ones(len(to_plot)) * np.max(to_plot), 'k--')
	ax[ii].plot(range(len(to_plot)), np.ones(len(to_plot)) * np.mean(to_plot), 'r--')
	ax[ii].plot(range(len(to_plot)), np.ones(len(to_plot)) * np.median(to_plot), 'b--')

plt.show()