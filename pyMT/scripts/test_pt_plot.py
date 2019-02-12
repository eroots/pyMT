import pyMT.data_structures as WSDS
import matplotlib.pyplot as plt
import numpy as np
import pyMT.utils as utils


def regulate_errors(data, raw_data, multiplier=2.5, fwidth=1, sites=None):
    use_log = False
    if sites is None:
        sites = raw_data.site_names
    smoothed = {site: {comp: [] for comp in ('PTXX', 'PTXY', 'PTYX', 'PTYY')}
                for site in sites}
    for site in sites:
        raw_site = raw_data.sites[site]
        data_site = data.sites[site]
        for comp in ('PTXX', 'PTXY', 'PTYX', 'PTYY'):
            # phi_data = np.array(data_site.phase_tensors[ii].comp for ii in range(data_site.NP))
            phi_raw = np.array([getattr(raw_site.phase_tensors[ii], comp) for ii in range(raw_site.NP)])
            smoothed_data = utils.geotools_filter(np.log10(raw_site.periods),
                                                  phi_raw,
                                                  fwidth=fwidth,
                                                  use_log=use_log)
            smoothed[site][comp] = smoothed_data
            for ii, p in enumerate(data_site.periods):
                ind = np.argmin(abs(raw_site.periods - p))
                max_error = np.max((multiplier * abs(getattr(data_site.phase_tensors[ii], comp) -
                                                    smoothed_data[ind]),
                                   getattr(data_site.phase_tensors[ii], comp + '_error')))
                # error_map[ii] = min([data_site.errmap[comp][ii],
                #                      np.ceil(max_error / (np.sqrt(p) * data_site.errors[comp][ii]))])
                # error_map[ii] =  np.ceil(max_error / (np.sqrt(p) * data_site.errors[comp][ii]))
                setattr(data.sites[site].phase_tensors[ii], comp + '_error', max_error)
            # data.sites[site].errmap[comp] = error_map
        # data.sites[site].apply_error_floor()
    return data, smoothed


# C:\Users\eric\phd\ownCloud\data\Regions\MetalEarth\swayze\R2south_4\pt
# datafile = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/R2south_4/pt/R2south_4d_Z_reset.dat'
# respfile = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/R2south_4/pt/R2South_fine_fwd_Z.dat'
# listfile = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/R2south_4c.lst'
datafile = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/' + \
           'swz_cull1/finish/pt/swz_cull1i_Z_extended_periods.dat'
respfile = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/finish/swz_finish.dat'
listfile = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/swz_cull1.lst'
data = WSDS.Data(datafile=datafile, listfile=listfile)
resp = WSDS.Data(datafile=respfile, listfile=listfile)
raw = WSDS.RawData(listfile=listfile)
data.inv_type = 6
error_in_degrees = 3
site = '18-swz080m'
data, smoothed = regulate_errors(data, raw, multiplier=1.5, fwidth=0.75, sites=None)
components = ('PTXY', 'PTYX', 'PTXX', 'PTYY')
for comp in components[:2]:
    phi_data = np.rad2deg(np.arctan(np.array([getattr(data.sites[site].phase_tensors[ii], comp)
                                              for ii in range(data.NP)])))
    phi_error = np.rad2deg(np.arctan(np.array([getattr(data.sites[site].phase_tensors[ii], comp + '_error')
                                               for ii in range(data.NP)])))
    phi_raw = np.rad2deg(np.arctan(np.array([getattr(raw.sites[site].phase_tensors[ii], comp)
                                             for ii in range(raw.sites[site].NP)])))
    phi_raw_error = np.rad2deg(np.arctan(np.array([getattr(raw.sites[site].phase_tensors[ii], comp + '_error')
                                                   for ii in range(raw.sites[site].NP)])))
    smoothed_data = np.rad2deg(np.arctan(smoothed[site][comp]))
    data_periods = np.log10(data.sites[site].periods)
    raw_periods = np.log10(raw.sites[site].periods)
    plt.errorbar(data_periods, phi_data, yerr=phi_error,
                 marker='o', linestyle='', color='gray', mec='k')
    plt.plot(raw_periods, phi_raw,
             marker='o', linestyle='', color='gray')
    plt.plot(raw_periods, smoothed_data, marker='', linestyle='-', color='k')
plt.show()
# data.write('C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/R2south_4/pt/R2south_4d_PT.dat')
data.write('C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/finish/pt/swz_cull1i_PT_extP.dat')
# used_components = ('ZXYR', 'ZXYI', 'ZYXR', 'ZYXI')
# for site in data.site_names:
#     for ii, p in enumerate(data.periods):
#         data.sites[site].data['ZXYR'][ii] = data.sites[site].phase_tensors[ii].phi[1, 1]
#         data.sites[site].data['ZXYI'][ii] = data.sites[site].phase_tensors[ii].phi[1, 0]
#         data.sites[site].data['ZYXR'][ii] = data.sites[site].phase_tensors[ii].phi[0, 1]
#         data.sites[site].data['ZYXI'][ii] = data.sites[site].phase_tensors[ii].phi[0, 0]
#         resp.sites[site].data['ZXYR'][ii] = resp.sites[site].phase_tensors[ii].phi[1, 1]
#         resp.sites[site].data['ZXYI'][ii] = resp.sites[site].phase_tensors[ii].phi[1, 0]
#         resp.sites[site].data['ZYXR'][ii] = resp.sites[site].phase_tensors[ii].phi[0, 1]
#         resp.sites[site].data['ZYXI'][ii] = resp.sites[site].phase_tensors[ii].phi[0, 0]
# components = ('ZXYR', 'ZXYI', 'ZYXR', 'ZYXI')
# data.reset_errors(error_floor=0.1, components=components)
# data.apply_outlier_map()
# for site in data.site_names:
#     data.sites[site].apply_error_floor()
# rms = {comp: 0 for comp in used_components}
# total_rms = 0
# for comp in used_components:
#     for site in data.site_names:
#         for ii in range(data.NP):
#             data.sites[site].used_error[comp][ii] = np.max((data.sites[site].used_error[comp][ii],
#                                                             np.tan(np.deg2rad(error_in_degrees))))
#             rms[comp] += ((data.sites[site].data[comp][ii] - resp.sites[site].data[comp][ii]) /
#                           data.sites[site].used_error[comp][ii]) ** 2
#     rms[comp] = np.sqrt(rms[comp] / (data.NP * data.NS))
#     total_rms += rms[comp] ** 2
# total_rms = np.sqrt(total_rms / len(used_components))


# with open('C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/R2south_4/pt/errors.dat', 'w') as f:
# with open('C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/finish/pt/errors.dat', 'w') as f:
#     for site in data.site_names:
#         for ii, p in enumerate(data.periods):
#             for comp in ('ZXYR', 'ZXYI', 'ZYXR', 'ZYXI'):
                # f.write('{:>14.7E}\n'.format(data.sites[site].used_error[comp][ii]))
# plt.errorbar(np.log10(data.periods), np.rad2deg(np.arctan(data.sites[site_name].data['ZXYR'])),
#              marker='o', linestyle='',
#              color='gray', yerr=np.rad2deg(np.arctan(data.sites[site_name].used_error['ZXYR'])))
# plt.errorbar(np.log10(data.periods), np.rad2deg(np.arctan(data.sites[site_name].data['ZXYI'])),
#              marker='o', linestyle='',
#              color='red', yerr=np.rad2deg(np.arctan(data.sites[site_name].used_error['ZXYI'])))
# plt.errorbar(np.log10(data.periods), np.rad2deg(np.arctan(data.sites[site_name].data['ZYXR'])),
#              marker='o', linestyle='',
#              color='green', yerr=np.rad2deg(np.arctan(data.sites[site_name].used_error['ZYXR'])))
# plt.errorbar(np.log10(data.periods), np.rad2deg(np.arctan(data.sites[site_name].data['ZYXI'])),
#              marker='o', linestyle='',
#              color='magenta', yerr=np.rad2deg(np.arctan(data.sites[site_name].used_error['ZYXI'])))
# plt.plot(np.log10(resp.periods), np.rad2deg(np.arctan(resp.sites[site_name].data['ZXYR'])),
#          marker='', linestyle='-', color='gray')
# plt.plot(np.log10(resp.periods), np.rad2deg(np.arctan(resp.sites[site_name].data['ZXYI'])),
#          marker='', linestyle='-', color='red')
# plt.plot(np.log10(resp.periods), np.rad2deg(np.arctan(resp.sites[site_name].data['ZYXR'])),
#          marker='', linestyle='-', color='green')
# plt.plot(np.log10(resp.periods), np.rad2deg(np.arctan(resp.sites[site_name].data['ZYXI'])),
#          marker='', linestyle='-', color='magenta')
# plt.show()
