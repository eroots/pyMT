import pyMT.data_structures as WSDS
import matplotlib.pyplot as plt
import numpy as np
import e_colours.colourmaps as cm


cmap = cm.jet_plus(64)
listfile = r'C:\Users\eric\phd\Kilauea\ConvertedEDIs\2018-517\allsites.lst'
# listfile = r'C:\Users\eric\phd\Kilauea\ConvertedEDIs\all\allsites.lst'
# datafile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Test_Models\dimensionality\synthLayer.data'
data = WSDS.RawData(listfile=listfile)
normalize = 0

periods = []
loc = []
X = []
Y = []
for jj, site in enumerate(data.site_names):
    for ii, p in enumerate(data.sites[site].periods):
        periods.append(p)
        loc.append(jj)
        X.append(-data.sites[site].data['TZXR'][ii])
        Y.append(-data.sites[site].data['TZYR'][ii])
arrows = np.transpose(np.array((X, Y)))
periods = np.log10(np.array(periods))
locs = np.array(loc)
xticks = np.arange(0, loc[-1], 24)
xtick_labels = [str(x) for x in np.arange(501, 531)]

lengths = np.sqrt(arrows[:, 0] ** 2 + arrows[:, 1] ** 2)
lengths[lengths == 0] = 1
arrows[lengths > 1, 0] /= lengths[lengths > 1]
arrows[lengths > 1, 1] /= lengths[lengths > 1]
max_length = np.sqrt((np.max(periods) -
                      np.min(periods)) ** 2 +
                     (np.max(loc) -
                      np.min(loc)) ** 2) / 10
if normalize:
    arrows /= np.transpose(np.tile(lengths, [2, 1]))
else:
    arrows /= np.max(lengths)
    arrows *= max_length / 100

plt.quiver(loc,
           periods,
           arrows[:, 1],
           arrows[:, 0],
           color='k',
           headwidth=1,
           width=0.0025,
           zorder=10)
plt.xlabel('Hour')
plt.ylabel(r'$\log_{10}$ Period (s)')
plt.show()
