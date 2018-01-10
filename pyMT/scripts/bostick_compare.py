import pyMT.data_structures as WSDS
import numpy as np
import matplotlib.pyplot as plt


dataset = WSDS.Dataset(modelfile=r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\wst\New\wst0_sens\wst0Inv5_model.02',
                       datafile=r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\wst\New\wst0_sens\wst0Inv5.data',
                       listfile=r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\wst\New\j2\cull5.lst',
                       datpath=r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\wst\New\j2')

sites = ['geo00_033', 'geo00_027', '98-1_116', '98-2_149']

rho = []
for ii, site in enumerate(sites):
    locs = dataset.data.sites[site].locations
    ix = np.argmin(abs(np.array(dataset.model.dx) - locs['X']))
    iy = np.argmin(abs(np.array(dataset.model.dy) - locs['Y']))
    rho.append(np.squeeze(dataset.model.vals[iy, ix, :]))

    plt.subplot(2, 2, ii + 1)
    plt.plot(np.log10(rho[ii]), np.log10(np.array(dataset.model.dz[1:]) / 1000))
    plt.ylim([plt.ylim()[-1], 0])
    plt.title(site)
plt.show()

