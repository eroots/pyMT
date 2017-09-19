import pyMT.data_structures as WSDS
import pyMT.utils as utils
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# raw = WSDS.RawData(r'C:\Users\eric\Documents\MATLAB' +
#                    r'\MATLAB\Inversion\Regions\abi-gren\New\j2\allsites.lst')
raw = WSDS.Data(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Test_Models\ztemTest\rotTest\evenMoreSites\evenMoreSitesSynth.data')

site_bost = {}
for site in raw.sites.values():
    bostick, depth = utils.compute_bost1D(site)[:2]
    if 'TZXR' in site.components:
        tzxr = site.data['TZXR']
        tzyr = site.data['TZYR']
    else:
        tzxr = site.data['ZXXR'] * 0
        tzyr = site.data['ZXYR'] * 0
    # site_bost.update({site.name: {'bostick': bostick, 'depth': depth,
    #                               'Long': site.locations['Long'],
    #                               'Lat': site.locations['Lat'],
    #                               'TZXR': tzxr, 'TZYR': tzyr}})
    site_bost.update({site.name: {'bostick': bostick, 'depth': depth,
                                  'X': site.locations['Y'],
                                  'Y': site.locations['X'],
                                  'TZXR': tzxr, 'TZYR': tzyr}})

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for site in site_bost.values():
    X = np.ones(site['depth'].shape) * site['X']
    Y = np.ones(site['depth'].shape) * site['Y']
    cax = ax.scatter(X, Y, site['depth'], c=site['bostick'])
plt.gca().invert_zaxis()
fig.colorbar(cax)
plt.set_cmap('jet_r')
plt.show()