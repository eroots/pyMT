import numpy as np
import pyMT.data_structures as DS
import matplotlib.pyplot as plt
import pyMT.utils as utils
from copy import deepcopy

# data_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wst_cullmantle3_LAMBERT_ZK_removed.dat'
# response_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wstZK_lastIter.dat'
data_file = 'E:/phd/NextCloud/data/synthetics/EM3DANI/wst/base_insidePers/wst_cullmantle3_LAMBERT_ZK_insidePers.dat'
# response_file = 'E:/phd/NextCloud/data/synthetics/EM3DANI/wst/aniso7/wstZK_aniso7j.resp'
save_path = 'E:/phd/NextCloud/Documents/ME_Transects/wst/rms_plots'
data = DS.Data(data_file)
# all_runs = ['j', 'k', 'n', 'o', 't']
all_runs = 'abcdefghijklmnopqrstuvwxy'
# response = DS.Data(response_file, listfile='E:/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst')
# for ii, site in enumerate(data.site_names):
#     response.sites[response.site_names[ii]]
# response.site_names = deepcopy(data.site_names)
multiplier = 1.2
fwidth = 0.85
# Only take the central block
idx = np.where((data.locations[:,0]>0) & (data.locations[:,0]<430000) & (data.locations[:,1]<100000))[0]
rm_idx = [x for x in range(data.NS) if x not in idx]
rm_sites = []
for ii in range(data.NS):
    if ii in rm_idx:
        rm_sites.append(data.site_names[ii])
data.remove_sites(sites=rm_sites)
data.site_names = sorted(data.site_names, key=lambda x: data.sites[x].locations['Y'])

# Generate smooth phi_max/min arrays
data_phi_max = np.zeros((data.NS, data.NP))
data_phi_min = np.zeros((data.NS, data.NP))
smooth_phi_max = np.zeros((data.NS, data.NP))
smooth_phi_min = np.zeros((data.NS, data.NP))
error_phi_max = np.zeros((data.NS, data.NP))
error_phi_min = np.zeros((data.NS, data.NP))
for ii, site in enumerate(data.site_names):
    data_phi_max[ii,:] = np.rad2deg(np.arctan(np.array([data.sites[site].phase_tensors[jj].phi_max for jj in range(data.NP)])))
    data_phi_min[ii,:] = np.rad2deg(np.arctan(np.array([data.sites[site].phase_tensors[jj].phi_min for jj in range(data.NP)])))


    # data_phi_max[ii,:] = utils.geotools_filter(np.log10(data.periods),
    #                                            np.rad2deg(np.arctan(phi_max)),
    #                                            use_log=False)
    # data_phi_min[ii,:] = utils.geotools_filter(np.log10(data.periods),
    #                                            np.rad2deg(np.arctan(phi_min)),
    #                                            use_log=False)
    smooth_phi_max[ii,:] = utils.geotools_filter(np.log10(data.periods),
                                                 data_phi_max[ii,:],
                                                 fwidth=fwidth, use_log=False)
    smooth_phi_min[ii,:] = utils.geotools_filter(np.log10(data.periods),
                                                 data_phi_min[ii,:],
                                                 fwidth=fwidth, use_log=False)
    error_phi_max[ii,:] = multiplier * abs(data_phi_max[ii,:] - smooth_phi_max[ii,:])
    error_phi_min[ii,:] = multiplier * abs(data_phi_min[ii,:] - smooth_phi_min[ii,:])

error_phi_max[error_phi_max<5] = 5
error_phi_min[error_phi_min<5] = 5

resp_phi_max = np.zeros((data.NS, data.NP, len(all_runs)))
resp_phi_min = np.zeros((data.NS, data.NP, len(all_runs)))
diff_phi_max = np.zeros((data.NS, data.NP, len(all_runs)))
diff_phi_min = np.zeros((data.NS, data.NP, len(all_runs)))
smoothdiff_phi_max = np.zeros((data.NS, data.NP, len(all_runs)))
smoothdiff_phi_min = np.zeros((data.NS, data.NP, len(all_runs)))
for kk, run_letter in enumerate(all_runs):
    response = DS.Data('E:/phd/NextCloud/data/synthetics/EM3DANI/wst/aniso7/wstZK_aniso7{}.resp'.format(run_letter),
                       listfile='E:/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst')
    response.remove_sites(sites=rm_sites)
    response.site_names = sorted(data.site_names, key=lambda x: data.sites[x].locations['Y'])

    for ii, site in enumerate(data.site_names):
       # Calculate differences
        resp_phi_min[ii,:,kk] = np.array([np.rad2deg(np.arctan(response.sites[site].phase_tensors[jj].phi_min)) for jj in range(data.NP)])
        resp_phi_max[ii,:,kk] = np.array([np.rad2deg(np.arctan(response.sites[site].phase_tensors[jj].phi_max)) for jj in range(data.NP)])

    diff_phi_min[:,:,kk] = resp_phi_min[:,:,kk] - data_phi_min[ii,:]
    diff_phi_max[:,:,kk] = resp_phi_max[:,:,kk] - data_phi_max[ii,:]
    smoothdiff_phi_min[:,:,kk] = resp_phi_min[:,:,kk] - smooth_phi_min[ii,:]
    smoothdiff_phi_max[:,:,kk] = resp_phi_max[:,:,kk] - smooth_phi_max[ii,:]
# Pseudo mask the points that are really bad
diff_phi_min[abs(diff_phi_min) > 20] = 5
diff_phi_max[abs(diff_phi_max) > 20] = 5
smoothdiff_phi_min[abs(smoothdiff_phi_min) > 20] = 5
smoothdiff_phi_max[abs(smoothdiff_phi_max) > 20] = 5

meandiff_phi_min = np.mean(np.abs(diff_phi_min), axis=0)
meandiff_phi_max = np.mean(np.abs(diff_phi_max), axis=0)
smoothmeandiff_phi_min = np.mean(np.abs(smoothdiff_phi_min), axis=0)
smoothmeandiff_phi_max = np.mean(np.abs(smoothdiff_phi_max), axis=0)

rmse_phi_min = np.sqrt(np.mean((diff_phi_min/np.repeat(error_phi_min[:,:,np.newaxis], len(all_runs), axis=2))**2, axis=0))
rmse_phi_max = np.sqrt(np.mean((diff_phi_max/np.repeat(error_phi_max[:,:,np.newaxis], len(all_runs), axis=2))**2, axis=0))

smoothrmse_phi_min = np.sqrt(np.mean((smoothdiff_phi_min/np.repeat(error_phi_min[:,:,np.newaxis], len(all_runs), axis=2))**2, axis=0))
smoothrmse_phi_max = np.sqrt(np.mean((smoothdiff_phi_max/np.repeat(error_phi_max[:,:,np.newaxis], len(all_runs), axis=2))**2, axis=0))

# # # Total (site and period averaged) rmse across each run
# plt.subplot(211)
# plt.bar(range(len(all_runs)),np.sqrt(np.mean(rmse_phi_max**2,axis=0)), width=0.2)
# plt.bar(np.arange(len(all_runs))+0.15,np.sqrt(np.mean(rmse_phi_min**2, axis=0)), width=0.2)
# plt.xticks(range(len(all_runs)),[x for x in all_runs])
# # Total (site and period averaged) rmse across each run against the smoothed data
# plt.subplot(212)
# plt.bar(range(len(all_runs)),np.sqrt(np.mean(smoothrmse_phi_max**2,axis=0)), width=0.2)
# plt.bar(np.arange(len(all_runs))+0.15,np.sqrt(np.mean(smoothrmse_phi_min**2, axis=0)), width=0.2)
# plt.xticks(range(len(all_runs)),[x for x in all_runs])
# plt.show()

# Average absolute difference at a specific period for each run
# for ii in range(data.NP):
#     idx_min = np.argsort(smoothmeandiff_phi_min[ii,:])
#     idx_max = np.argsort(smoothmeandiff_phi_max[ii,:])
#     plt.subplot(2,4,ii+1)
#     plt.bar(range(len(all_runs)), smoothmeandiff_phi_min[ii,:], width=0.2)
#     plt.bar(np.arange(len(all_runs))+0.15,smoothmeandiff_phi_max[ii,:], width=0.2)
#     plt.xticks(range(len(all_runs)),[x for x in all_runs])
#     plt.title('{}, {}, {}, {}, {}, {}'.format(all_runs[idx_min[0]], all_runs[idx_min[1]], all_runs[idx_min[2]],
#                                               all_runs[idx_max[0]], all_runs[idx_max[1]], all_runs[idx_max[2]]))
# plt.show()

# Site averaged RMS at a specific period for each run
for ii in range(data.NP):
    idx_min = np.argsort(rmse_phi_min[ii,:])
    idx_max = np.argsort(rmse_phi_max[ii,:])
    plt.subplot(2,4,ii+1)
    plt.bar(range(len(all_runs)), rmse_phi_min[ii,:], width=0.2)
    plt.bar(np.arange(len(all_runs))+0.15,rmse_phi_max[ii,:], width=0.2)
    plt.xticks(range(len(all_runs)),[x for x in all_runs])
    plt.title('{}, {}, {}, {}, {}, {}'.format(all_runs[idx_min[0]], all_runs[idx_min[1]], all_runs[idx_min[2]],
                                              all_runs[idx_max[0]], all_runs[idx_max[1]], all_runs[idx_max[2]]))
plt.show()


# width=0.05
# for ii in range(len(all_runs)):
#     plt.subplot(211)
#     plt.bar(np.log10(data.periods)+width*ii, smoothmeandiff_phi_min[:, ii], width=width, edgecolor='k', label=all_runs[ii])
#     plt.subplot(212)
#     plt.bar(np.log10(data.periods)+width*ii, smoothmeandiff_phi_max[:, ii], width=width, edgecolor='k', label=all_runs[ii])
# plt.legend()
# plt.show()


# for ii in range(data.NP):
    # plt.subplot(4,2,ii+1)
    # plt.hist(abs(diff_phi_min[:,ii,:]), range(0,10), density=True, edgecolor='black', linewidth='1.5')
# plt.show()