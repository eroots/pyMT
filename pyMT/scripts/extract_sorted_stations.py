import pyMT.data_structures as DS
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


grid_dims = 40
write_ew = 1
write_ns = 1
plot_it = 1
list_outpath = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/comsol/study15/'
data = DS.RawData('E:/phd/NextCloud/data/Regions/MetalEarth/wst/comsol/study15/all.lst')

ew_sites = sorted(data.site_names, key=lambda x: (data.sites[x].locations['Lat'], data.sites[x].locations['Long']))
data.site_names = ew_sites
data.locations = data.get_locs(mode='latlong')

if write_ew:
	for ii in range(grid_dims):
		with open(list_outpath+'EW_line{}.lst'.format(ii), 'w') as f:
			f.write('{}\n'.format(grid_dims))
			for s in data.site_names[ii*grid_dims:(ii+1)*grid_dims]:
				f.write('{}\n'.format(s))


temp_sites = deepcopy(ew_sites)
ns_sites = []
for ii in range(grid_dims):
	for jj in range(grid_dims):
		ns_sites.append(temp_sites[jj*40+ii])

data_ns = deepcopy(data)
data_ns.site_names = ns_sites
data_ns.locations = data_ns.get_locs(mode='latlong')
if write_ns:
	for ii in range(grid_dims):
		with open(list_outpath+'NS_line{}.lst'.format(ii), 'w') as f:
			f.write('{}\n'.format(grid_dims))
			for s in data_ns.site_names[ii*grid_dims:(ii+1)*grid_dims]:
				f.write('{}\n'.format(s))

colours = ['k', 'b', 'g', 'y', 'r']
if plot_it:
	for ii in range(5):
		plt.plot(data.locations[(ii*5)*40:(ii*5+1)*40, 1],
				 data.locations[(ii*5)*40:(ii*5+1)*40, 0],
				 colours[ii]+'.')
	plt.show()
	for ii in range(5):
		plt.plot(data_ns.locations[(ii*5)*40:(ii*5+1)*40, 1],
				 data_ns.locations[(ii*5)*40:(ii*5+1)*40, 0],
				 colours[ii]+'.')

	plt.show()