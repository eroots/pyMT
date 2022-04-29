import pyMT.data_structures as DS
import numpy as np
import matplotlib.pyplot as plt
import os
import pyMT.utils as utils
from matplotlib.backends.backend_pdf import PdfPages


def calculate_response(thickness, rho, period_range):
        scale = 1 / (4 * np.pi / 10000000)
        mu = 4 * np.pi * 1e-7
        periods = np.logspace(np.log10(period_range[0]),
                              np.log10(period_range[1]),
                              80)
        omega = 2 * np.pi / periods
        # d = np.cumsum(thickness + [100000])
        d = np.array(thickness + [1000000]) * 1000
        r = rho
        cond = 1 / np.array(r)
        # r = 1 / np.array(r)
        Z = np.zeros(len(periods), dtype=complex)
        rhoa = np.zeros(len(periods))
        phi = np.zeros(len(periods))
        for nfreq, w in enumerate(omega):
            prop_const = np.sqrt(1j*mu*cond[-1] * w)
            C = np.zeros(len(r), dtype=complex)
            C[-1] = 1 / prop_const
            if len(d) > 1:
                for k in reversed(range(len(r) - 1)):
                    prop_layer = np.sqrt(1j*w*mu*cond[k])
                    k1 = (C[k+1] * prop_layer + np.tanh(prop_layer * d[k]))
                    k2 = ((C[k+1] * prop_layer * np.tanh(prop_layer * d[k])) + 1)
                    C[k] = (1 / prop_layer) * (k1 / k2)
        # #         k2 = np.sqrt(1j*omega[nfreq]*C*mu0/r[k+1]);
        #         g = (g*k2+k1*np.tanh(k1*d[k]))/(k1+g*k2*np.tanh(k1*d[k]));
            Z[nfreq] = 1j * w * mu * C[0]

        # rhoa = 1/omega*np.abs(Z)**2;
        phi = np.angle(Z, deg=True)
        det = np.sqrt(Z * Z)
        rhoa = np.abs(det * np.conj(det) * periods / (mu * 2 * np.pi))
        return rhoa, phi, Z, periods, d, r

# path = 'E:/phd/NextCloud/data/Regions/TTZ/1D_inversions/'
# data_file = 'E:/phd/NextCloud/data/Regions/TTZ/full_run/ttz_full_cull1.dat'
# model_file = 'E:/phd/NextCloud/data/Regions/TTZ/full_run/ttz_hs2500.rho'
# output_path = 'E:/phd/NextCloud/data/Regions/TTZ/1D_inversions/'
path = 'E:/phd/NextCloud/data/Regions/snorcle/1D_inversions/normal/single_site_ssq/'
data_file = 'E:/phd/NextCloud/data/Regions/snorcle/j2/jformat-0TN/j2edi/ffmt_output/renamed/sorted_cull1b.lst'
model_file = 'E:/phd/NextCloud/data/Regions/snorcle/cull1/sno_cull1_hs100_nest.model'
output_path = 'E:/phd/NextCloud/data/Regions/snorcle/1D_inversions/normal/'
data = DS.RawData(data_file)
use_rho = 1
use_phase = 1
use_component = 'ssq'
write_model = True
write_pdf = False
output_model = output_path + 'sno_LogAvg1D_nest'
# model = DS.Model(model_file)
rho = {site: utils.compute_rho(data.sites[site], calc_comp=use_component)[0] for site in data.site_names}
rhoxy_err = {site: utils.compute_rho(data.sites[site], calc_comp='xy', errtype='used_error')[1] for site in data.site_names}
rhoyx_err = {site: utils.compute_rho(data.sites[site], calc_comp='yx', errtype='used_error')[1] for site in data.site_names}
rho_err = {site: (rhoxy_err[site] + rhoyx_err[site]) / 2 for site in data.site_names}
phase = {site: utils.compute_phase(data.sites[site], calc_comp=use_component)[0] for site in data.site_names}
rho_smooth = {site: utils.geotools_filter(data.sites[site].periods, rho[site], fwidth=0.65) for site in data.site_names}
phase_smooth = {site: utils.geotools_filter(data.sites[site].periods, phase[site], fwidth=0.65) for site in data.site_names}
# use_dz = [0] + list(np.logspace(1, 6, 200))
# use_dz = [0] + list(np.logspace(0, 5.5, 200))
use_dz = [0] + list(np.logspace(0, 6, 300))
num_model = 0
model = np.zeros((len(use_dz), data.NS))
# use_sites = data.site_names
use_sites = ['average_site']
for site in use_sites:
    iters = []
    if site not in ('CS_site3', 'CS_site14'):      
    # if site not in ['ttz14_204', 'ttz14_208']:
        # files = [file for file in os.listdir(path + site) if (file.endswith('iter') and 'rho' in file)] # and file.startswith('CS'))]
        files = [file for file in os.listdir(path + site) if (file.endswith('iter'))]# and file.startswith('CS'))]
        last_iters = max([int(f.replace('_', '').replace('.iter', '')[-1]) for f in files])
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
            if misfit < 0.51:
                model[:, num_model] = np.array(([10 ** float(x.strip()) for x in f.read().split()]))
        num_model += 1
# stop
# Remove zero columns
model = model[:, ~np.all(model==0, axis=0)]
model = 10 ** np.mean(np.log10(model), axis=1)
# model = np.mean(model, axis=1)

all_periods = []
for site in data.site_names:
    all_periods += list(data.sites[site].periods)
all_periods = sorted(list(set(all_periods)))
rho_resp, phase_resp, Z, periods, d, r = calculate_response(list(np.diff(use_dz)/1000), list(model), [all_periods[0], all_periods[-1]])

# with PdfPages('E:/phd/NextCloud/data/Regions/MetalEarth/TTZ/1D_inversions/logmean_model_response.pdf') as pdf:
if write_pdf:
    with PdfPages('E:/phd/NextCloud/data/Regions/snorcle/1D_inversions/normal/single-site-{}_logmean_model_response.pdf'.format(use_component)) as pdf:
        for site in data.sites:
            fig = plt.figure(figsize=(12, 16))
            plt.gcf().add_subplot(2,1,1)
            plt.loglog(use_dz, model)
            # plt.ylim([10, 50000])
            plt.xlabel('Depth (m)')
            plt.ylabel('Resistivity (ohm-m)')
            plt.title(site)
            plt.gcf().add_subplot(2,1,2)
            if use_rho and use_phase:
                plt.plot((data.sites[site].periods), (rho[site]), mec='r', color='r', marker='v', linestyle='', zorder=0, label='RawData Rho')
                # plt.loglog(data.sites[site].periods, rho_smooth[site], 'r-', zorder=2, label='Inverted Rho Data')
                plt.loglog(periods, rho_resp, 'r--', zorder=1, label='Model Response - Rho')
                plt.ylabel('{} App. Resistivity (ohm-m)'.format(use_component))
                ax2 = plt.gca().twinx()
                ax2.plot((data.sites[site].periods), (phase[site]), mec='b', color='b', marker='^', linestyle='', zorder=0, label='RawData Phase')
                # ax2.semilogx(data.sites[site].periods, phase_smooth[site], 'b-', zorder=2, label='Inverted Phase Data')
                ax2.semilogx(periods, phase_resp, 'b--', zorder=1, label='Model Response - Phase')
                ax2.set_ylabel('Phase (Degrees)')
            elif use_rho:
                plt.errorbar((data.sites[site].periods), (rho[site]), mec='k', color='k', marker='.', linestyle='', xerr=None, yerr=rho_err[site], zorder=0, label='RawData')
                plt.loglog(data.sites[site].periods, rho_smooth[site], zorder=2, label='Smoothed (Inverted) Data')
                plt.loglog(periods, rho_resp, 'r--', zorder=1, label='Model Response - Rho')
                plt.ylabel('{} App. Resistivity (ohm-m)'.format(use_component))
            elif use_phase:
                plt.errorbar((data.sites[site].periods), (phase[site]), mec='k', color='k', marker='.', linestyle='', xerr=None, yerr=pha_err[site], zorder=0, label='RawData')
                plt.semilogx(data.sites[site].periods, phase_smooth[site], zorder=2, label='Smoothed (Inverted) Data')
                plt.semilogx(periods, phase_resp, 'b--', zorder=1, label='Model Response - Phase')
                plt.ylabel('Phase (Degrees)')
            plt.xlabel('Period (s)')

            # plt.loglog(data.sites[site].periods, rho_smooth[site], zorder=2, label='Smoothed (Inverted) Data')
            # plt.loglog(periods, rho_resp, '--', zorder=1, label='Model Response')
            # plt.errorbar((data.sites[site].periods), (rho[site]), mec='k', color='k', marker='.', linestyle='', xerr=None, yerr=rho_err[site], zorder=0, label='RawData')
            # plt.ylim([10, 50000])
            # plt.xlabel('Period (s)')
            # plt.ylabel('Det App. Resistivity (ohm-m)')
            plt.legend()
            ax2.legend()
            # plt.title('Misfit: {:>5.3f}'.format(misfit))
            # plt.show()
            pdf.savefig(fig)
            plt.close()

if write_model:
    mod3d = DS.Model(model_file)

    idx_air = mod3d.vals > mod3d.RHO_AIR / 10
    idx_ocean = mod3d.vals < mod3d.RHO_OCEAN + 0.1
    for iz in range(mod3d.nz):
        idx = np.argmin(np.abs(use_dz - mod3d.dz[iz]))
        mod3d.vals[:, :, iz] = model[idx]
    mod3d.vals[idx_air] = mod3d.RHO_AIR
    mod3d.vals[idx_ocean] = mod3d.RHO_OCEAN
    mod3d.write(output_model)
# mod3d.write('E:/phd/NextCloud/data/Regions/MetalEarth/swayze/R2Southeast_3/3D/rot330/SWZAvgLog1D_large.model')