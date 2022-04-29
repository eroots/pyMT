import pyMT.data_structures as DS
import numpy as np
import matplotlib.pyplot as plt
import os
import pyMT.utils as utils
from matplotlib.backends.backend_pdf import PdfPages


path = 'E:/phd/NextCloud/data/Regions/snorcle/1D_inversions/normal/allsites_det/'
data_file = 'E:/phd/NextCloud/data/Regions/snorcle/j2/jformat-0TN/j2edi/ffmt_output/renamed/sorted_cull1.lst'
# model_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/R2Southeast_3/full_run/ttz_hs2500.rho'
output_path = 'E:/phd/NextCloud/data/Regions/snorcle/1D_inversions/normal/'
# data = DS.Data(data_file)
data = DS.RawData(data_file)
used_component = 'det'
use_rho = 1
use_phase = 1
low_cut = 0.0001
high_cut = 300
for site in data.site_names:
    s = data.sites[site]
    remove_periods = s.periods[(s.periods < low_cut) + (s.periods > high_cut)]
    data.sites[site].remove_periods(periods=remove_periods)
# model = DS.Model(model_file)
rho = {site: utils.compute_rho(data.sites[site], calc_comp=used_component)[0] for site in data.site_names}
rhoxy_err = {site: utils.compute_rho(data.sites[site], calc_comp='xy', errtype='used_error')[1] for site in data.site_names}
rhoyx_err = {site: utils.compute_rho(data.sites[site], calc_comp='yx', errtype='used_error')[1] for site in data.site_names}
rho_err = {site: (rhoxy_err[site] + rhoyx_err[site]) / 2 for site in data.site_names}
phasexy_err = {site: utils.compute_phase(data.sites[site], calc_comp='xy', errtype='used_error')[1] for site in data.site_names}
phaseyx_err = {site: utils.compute_phase(data.sites[site], calc_comp='yx', errtype='used_error')[1] for site in data.site_names}
pha = {site: utils.compute_phase(data.sites[site], calc_comp=used_component)[0] for site in data.site_names}
pha_err = {site: (phasexy_err[site] + phaseyx_err[site]) / 2 for site in data.site_names}
rho_smooth = {site: utils.geotools_filter(data.sites[site].periods, rho[site], fwidth=0.65) for site in data.site_names}
pha_smooth = {site: utils.geotools_filter(data.sites[site].periods, pha[site], fwidth=0.65) for site in data.site_names}
# use_dz = [0] + list(np.logspace(0, 5, 200))
# use_dz = [0] + list(np.logspace(0, 5.5, 200))
use_dz = [0] + list(np.logspace(0, 6, 300))
use_sites = [s for s in data.site_names if s not in ['18-swz108a']]
with PdfPages('E:/phd/NextCloud/data/Regions/snorcle/1D_inversions/normal/allsites_det-rhopha.pdf') as pdf:
    for site in use_sites[:10]:
        # if site[-3:] not in ('106', '201','205','207','208','210','211'):
        # if site[-3:] not in ['204',]:
            # model = np.zeros(201)
        files = [file for file in os.listdir(path + site) if (file.endswith('iter')) and 'rho' in file] # and file.startswith('CS'))]
        iters = []
        for f in files:
            try:
                iters.append(int(f[-7:-5]))
            except ValueError:
                iters.append(int(f[-6:-5]))
        last_iters = max(iters)
        # with open(path + site + '/' + '_' + str(last_iters) + '.iter', 'r') as f:
        with open(path + site + '/' + site + '_rho_' + str(last_iters) + '.iter', 'r') as f:
            for ii in range(17):
                f.readline()        
            misfit = float(f.readline().split()[-1].strip())
            for ii in range(2):
                f.readline()
            # for ii in range(201):
            model = np.array(([10 ** float(x.strip()) for x in f.read().split()]))
        # resp = DS.Data(path + file + '/CS_site1_' + str(last_iters) + '.resp')
        # with open(path + site + '/' + '_' + str(last_iters) + '.resp', 'r') as f:
        with open(path + site + '/' + site + '_rho_' + str(last_iters) + '.resp', 'r') as f:
            lines = f.readlines()
            all_data = np.array([float(x.split()[6].strip()) for x in lines[data.sites[site].NP+11:]])

        fig = plt.figure(figsize=(12, 16))
        plt.gcf().add_subplot(2,1,1)
        plt.loglog(use_dz, model)
        plt.xlabel('Depth (m)')
        plt.ylabel('Resistivity (ohm-m)')
        # plt.ylim([100, 100000])
        plt.title(site)
        plt.gcf().add_subplot(2,1,2)
        if use_rho and use_phase:
            plt.plot((data.sites[site].periods), (rho[site]), mec='k', color='k', marker='.', linestyle='', zorder=0, label='RawData Rho')
            plt.loglog(data.sites[site].periods, rho_smooth[site], 'r-', zorder=2, label='Inverted Rho Data')
            plt.loglog(data.sites[site].periods, all_data[::2], 'r--', zorder=1, label='Model Response - Rho')
            plt.ylabel('{} App. Resistivity (ohm-m)'.format(used_component))
            ax2 = plt.gca().twinx()
            ax2.plot((data.sites[site].periods), (pha[site]), mec='k', color='k', marker='.', linestyle='', zorder=0, label='RawData Phase')
            ax2.semilogx(data.sites[site].periods, pha_smooth[site], 'b-', zorder=2, label='Inverted Phase Data')
            ax2.semilogx(data.sites[site].periods, all_data[1::2], 'b--', zorder=1, label='Model Response - Phase')
            ax2.set_ylabel('Phase (Degrees)')
        elif use_rho:
            plt.errorbar((data.sites[site].periods), (rho[site]), mec='k', color='k', marker='.', linestyle='', xerr=None, yerr=rho_err[site], zorder=0, label='RawData')
            plt.loglog(data.sites[site].periods, rho_smooth[site], zorder=2, label='Smoothed (Inverted) Data')
            plt.loglog(data.sites[site].periods, all_data, 'r--', zorder=1, label='Model Response - Rho')
            plt.ylabel('{} App. Resistivity (ohm-m)'.format(used_component))
        elif use_phase:
            plt.errorbar((data.sites[site].periods), (pha[site]), mec='k', color='k', marker='.', linestyle='', xerr=None, yerr=pha_err[site], zorder=0, label='RawData')
            plt.semilogx(data.sites[site].periods, pha_smooth[site], zorder=2, label='Smoothed (Inverted) Data')
            plt.semilogx(data.sites[site].periods, all_data, 'b--', zorder=1, label='Model Response - Phase')
            plt.ylabel('Phase (Degrees)')
        plt.xlabel('Period (s)')
        
        # plt.semilogx(data.periods, pha_smooth[site], zorder=2)
        # plt.semilogx(data.periods, all_data, '--', zorder=1)
        # plt.errorbar((data.sites[site].periods), (rho[site]), mec='k', color='k', marker='.', linestyle='', xerr=None, yerr=pha_err[site], zorder=0, label='RawData')
        # plt.ylim([100, 100000])
        # plt.ylim([0, 90])
        plt.title('Misfit: {:>5.3f}'.format(misfit))
        plt.legend()
        # plt.show()
        pdf.savefig(fig)
        plt.close()

