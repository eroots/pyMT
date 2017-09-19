import pyMT.data_structures as WSDS
import pyMT.utils as utils
import matplotlib.pyplot as plt
import numpy as np

listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\aften\j2\deci_4.lst'
raw = WSDS.RawData(listfile)
data = WSDS.Data(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\aften\deci0_4\deci_4_1.data',
                 listfile=listfile)
site = 'l6_351'

raw_site = raw.sites[site]
data_site = data.sites[site]

fig = plt.figure()
fig.add_subplot(121)
plt.errorbar(np.log10(data_site.periods), np.sqrt(data_site.periods) * data_site.data['ZXYR'],
             xerr=None, yerr=np.sqrt(data_site.periods) * data_site.used_error['ZXYR'],
             marker='o', mec='k', linestyle='')
plt.errorbar(np.log10(raw_site.periods), np.sqrt(raw_site.periods) * raw_site.data['ZXYR'],
             xerr=None, yerr=None, marker='o', linestyle='')
error_map = np.zeros(data_site.data['ZXYR'].shape)
for comp in ['ZXYR']:  # data_site.components:
    smoothed_data = utils.geotools_filter(np.log10(raw_site.periods),
                                          np.sqrt(raw_site.periods) * raw_site.data[comp],
                                          fwidth=1)
    for ii, p in enumerate(data_site.periods):
        ind = np.argmin(abs(raw_site.periods - p))
        max_error = 5 * abs(np.sqrt(p) * data_site.data[comp][ii] - smoothed_data[ind])
        error_map[ii] = min([data_site.errmap[comp][ii],
                             np.ceil(max_error / (np.sqrt(p) * data_site.errors[comp][ii]))])
        # new_errors[ii] = np.sqrt(p) * data_site.errors[comp][ii] * error_map
data_site.errmap['ZXYR'] = error_map
data_site.calculate_used_errors()
plt.plot(np.log10(raw_site.periods), smoothed_data)

fig.add_subplot(122)

plt.errorbar(np.log10(data_site.periods), np.sqrt(data_site.periods) * data_site.data['ZXYR'],
             xerr=None, yerr=np.sqrt(data_site.periods) * data_site.used_error['ZXYR'],
             marker='o', mec='k', linestyle='')
plt.errorbar(np.log10(raw_site.periods), np.sqrt(raw_site.periods) * raw_site.data['ZXYR'],
             xerr=None, yerr=None, marker='o', linestyle='')
plt.plot(np.log10(raw_site.periods), smoothed_data)
plt.show()
