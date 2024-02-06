import pyMT.data_structures as DS
import matplotlib.pyplot as plt
import numpy as np


data = DS.Data('E:/phd/Nextcloud/data/Regions/MetalEarth/rouyn/rou_north_all.dat')
save_fig = 1
save_path = 'E:/phd/Nextcloud/data/Regions/MetalEarth/rouyn/rapolai_2d/'
all_beta = []
for site in data.site_names:
    beta = np.array([abs(np.rad2deg(np.arctan(data.sites[site].phase_tensors[ii].beta))) for ii in range(data.NP)])
    all_beta.append(beta)
    #beta[beta > 10] = 10
    plt.semilogx(data.periods, beta, '.', label=site)
    plt.plot([0, data.periods[-1]], [3, 3], 'k--', label='3 degrees')
    plt.plot([0, data.periods[-1]], [6, 6], 'r--', label='6 degrees')
    plt.xlabel('Period (s)')
    plt.ylabel('Beta (degrees)')
    plt.legend()
    if save_fig:
        plt.savefig(save_path + site + '_beta.png')
    plt.close()
    if not save_fig:    
        plt.show()
