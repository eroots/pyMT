import numpy as np
import pyMT.data_structures as DS
import matplotlib.pyplot as plt
import pyMT.utils as utils

listfile = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst'
data_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wst_cullmantle3_LAMBERT_ZK_removed.dat'
response_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/feature_tests/wstZK-C2_resistor_resp.dat'

# response_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/feature_tests/wstZK_resp.dat'
# response_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/feature_tests/wstZK_C5a-10000ohm_NLCG_000.dat'
# response_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/depth_test/wstZK_depthTest_NLCG_000.dat'
# response_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wstZK_lastIter.dat'
save_path = 'E:/phd/NextCloud/Documents/ME_Transects/wst/rms_plots'
dataset = DS.Dataset(listfile=listfile, datafile=data_file, responsefile=response_file)

dataset.sort_sites(order='south-north')

diag = []
off_diag = []
impedance = []
tipper = []

for site in dataset.data.site_names:
    diag.append(utils.rms(np.array([dataset.rms['Station'][site]['ZXXR'],
                        dataset.rms['Station'][site]['ZXXI'],
                        dataset.rms['Station'][site]['ZYYR'],
                        dataset.rms['Station'][site]['ZYYI']])))
    off_diag.append(utils.rms(np.array([dataset.rms['Station'][site]['ZXYR'],
                            dataset.rms['Station'][site]['ZXYI'],
                            dataset.rms['Station'][site]['ZYXR'],
                            dataset.rms['Station'][site]['ZYXI']])))
    impedance.append(utils.rms(np.array([dataset.rms['Station'][site]['ZXYR'],
                            dataset.rms['Station'][site]['ZXYI'],
                            dataset.rms['Station'][site]['ZYXR'],
                            dataset.rms['Station'][site]['ZYXI'],
                            dataset.rms['Station'][site]['ZXXR'],
                            dataset.rms['Station'][site]['ZXXI'],
                            dataset.rms['Station'][site]['ZYYR'],
                            dataset.rms['Station'][site]['ZYYI']])))
    tipper.append(utils.rms(np.array([dataset.rms['Station'][site]['TZXR'],
                          dataset.rms['Station'][site]['TZXI'],
                          dataset.rms['Station'][site]['TZYR'],
                          dataset.rms['Station'][site]['TZYI']])))

impedance = np.array(impedance)
tipper = np.array(tipper)
idx = tipper < 0.01
idx2 = tipper >= 0.01
tipper = np.hstack((tipper[idx2], tipper[idx]))
impedance = np.hstack((impedance[idx2], impedance[idx]))
tipper[tipper < 0.01] = np.nan
total = np.hstack((impedance, tipper))

p_impedance = []
p_off_diag = []
p_diag = []
p_tipper = []

for ii in range(dataset.data.NP):
    p_diag.append(utils.rms(np.array([dataset.rms['Period']['ZXXR'][ii],
                          dataset.rms['Period']['ZXXI'][ii],
                          dataset.rms['Period']['ZYYR'][ii],
                          dataset.rms['Period']['ZYYI'][ii]])))
    p_off_diag.append(utils.rms(np.array([dataset.rms['Period']['ZXXR'][ii],
                              dataset.rms['Period']['ZXYI'][ii],
                              dataset.rms['Period']['ZYXR'][ii],
                              dataset.rms['Period']['ZYXI'][ii]])))
    p_impedance.append(utils.rms(np.array([dataset.rms['Period']['ZXXR'][ii],
                          dataset.rms['Period']['ZXXI'][ii],
                          dataset.rms['Period']['ZYYR'][ii],
                          dataset.rms['Period']['ZYYI'][ii],
                          dataset.rms['Period']['ZXXR'][ii],
                          dataset.rms['Period']['ZXYI'][ii],
                          dataset.rms['Period']['ZYXR'][ii],
                          dataset.rms['Period']['ZYXI'][ii]])))

    p_tipper.append(utils.rms(np.array([dataset.rms['Period']['TZXR'][ii],
                            dataset.rms['Period']['TZXI'][ii],
                            dataset.rms['Period']['TZYR'][ii],
                            dataset.rms['Period']['TZYI'][ii]])))

p_tipper = np.array(p_tipper)
p_tipper[p_tipper < 0.01] = np.nan

p_tipper_mag = np.zeros(dataset.data.NP)
p_tipper_ang = np.zeros(dataset.data.NP)
for site in dataset.data.site_names:
    d_abs = np.sqrt(dataset.data.sites[site].data['TZXR']**2 +
                    dataset.data.sites[site].data['TZXI']**2)
    r_abs = np.sqrt(dataset.response.sites[site].data['TZXR']**2 +
                    dataset.response.sites[site].data['TZXI']**2)
    d_ang = np.angle(dataset.data.sites[site].data['TZXR'] +
                     1j*dataset.data.sites[site].data['TZXI'], deg=True)
    r_ang = np.angle(dataset.response.sites[site].data['TZXR'] +
                     1j*dataset.response.sites[site].data['TZXI'], deg=True)
    for ii in range(dataset.data.NP):
        p_tipper_mag[ii] += (np.abs((d_abs[ii] - r_abs[ii])))**2
        p_tipper_ang[ii] += (np.abs((d_ang[ii] - r_ang[ii])))**2

p_tipper_mag = np.sqrt(p_tipper_mag / dataset.data.NS / 0.05)
p_tipper_ang = np.sqrt(p_tipper_ang / dataset.data.NS / 5)

plt.subplot(2,1,1)
# plt.plot(diag, 'b-', label='Diagonal')
# plt.plot(off_diag, 'r-', label='Off-Diagonal')
plt.plot(impedance, 'b-', label='Impedance')
plt.plot(tipper, 'g-', label='Tipper')
plt.plot(range(len(impedance)), np.ones(len(impedance)) * utils.rms(impedance), 'b--')
plt.plot(range(len(impedance)), np.ones(len(impedance)) * utils.rms(tipper), 'g--')
plt.legend()
plt.xlabel('Station #')
plt.ylabel('RMS Misfit')

plt.subplot(2,1,2)
# plt.semilogx(dataset.data.periods, p_diag, 'b-', label='Diagonal')
# plt.semilogx(dataset.data.periods, p_off_diag, 'r-', label='Off-Diagonal')
plt.semilogx(dataset.data.periods, p_impedance, 'b-', label='Impedance')
plt.semilogx(dataset.data.periods, p_tipper, 'g-', label='Tipper')
plt.legend()
plt.xlabel('Log10 Period (s)')
plt.ylabel('RMS Misfit')

# plt.subplot(3,1,3)
# plt.semilogx(dataset.data.periods, p_tipper_mag, 'b-', label='Tipper Magnitude')
# plt.semilogx(dataset.data.periods, p_tipper_ang, 'r-', label='Tipper Angle')
# plt.legend()

plt.show()