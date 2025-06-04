import numpy as np
import matplotlib.pyplot as plt


def rho(x):
    return 1 / x

def sigma(x):
    return 1 / x

def archies(sig1 ,sig2 ,m):
    prop = np.arange(0, 1.0, 0.001)
    prop[np.isnan(prop)] = 0
    p = np.log(1 - prop ** m) / np.log(1 - prop)
    sig_eff = sig1 * (1-prop) ** p + sig2*prop**m
    return sig_eff, prop

def hs_bounds(sig1, sig2):
    prop = np.arange(0, 1.0, 0.001)
    prop[np.isnan(prop)] = 0
    sig_U = sig2 + ((1 - prop) / (1 / (sig1-sig2) + (prop/(3*sig2))))
    sig_L = sig1 + (prop / ((1 / (sig2-sig1)) + ((1-prop)/(3*sig1))))
    return sig_U, sig_L

fig, ax = plt.subplots(figsize=(16, 12))
do_archies = True
do_hs = True
colors = ['b','r','c','g','y','m']
ii = 0
# for sig1 in [0.001, 0.0001, 0.00001]:
for sig1 in [0.0001]:
    for sig2 in [10, 1, 0.1]:
    # for sig2 in [10]:
        if do_archies:
            sig_eff, prop = archies(sig1, sig2, 1.)
            ax.semilogy(prop, sig_eff, colors[ii]+'-',# label=r'$\rho_{host}$: ' + 
                        # str(round(1/sig1)) + r' $\Omega$-m' + '\n' + 
                        label=r'$\rho_{2}$: ' + str(round(1/sig2, 1)) + r' $\Omega$m')
        if do_hs:
            sig_U, sig_L = hs_bounds(sig1, sig2)
            ax.semilogy(prop, sig_U, colors[ii]+'--', #label=r'HS Upper Bound, ' + 
                        # str(round(1/sig1)) + r' $\Omega$m' + '\n' + 
                        label=r'$\rho_{2}$: ' + str(round(1/sig2, 1)) + r' $\Omega$m')
            ax.semilogy(prop, sig_L, colors[ii]+'.-', #label=r'HS Lower Bound, ' + 
                        #str(round(1/sig1)) + r' $\Omega$m' + '\n' +
                        label=r'$\rho_{2}$: ' + str(round(1/sig2, 1)) + r' $\Omega$m',
                        markevery=10)
        ii += 1
plt.ylabel('Conductivity (S/m)', fontsize=16)
secax = ax.secondary_yaxis('right', functions=(sigma, rho))
# secax.set_ylabel(r'Resistivity  ($\Omega$-m)', fontsize=14)
plt.xlabel('Proportion of Phase 2', fontsize=16)
plt.semilogy(range(-1, 2, 1), [1 / 300] * 3, 'k--')#, label=r'300 $\Omega$-m')
plt.semilogy(range(-1, 2, 1), [1 / 50] * 3, 'k--')#, label=r'50 $\Omega$-m')
# plt.semilogy(prop, [1 / 10] * len(prop), 'k--', label=r'10 \Omega-m')
ncol = 2*do_hs + 1*do_archies
if ncol == 3:
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = handles[::3] + handles[1::3] + handles[2::3]
    new_labels = labels[::3] + labels[1::3] + labels[2::3]
    plt.legend(new_handles, new_labels, fontsize='large', ncol=ncol)
else:
    plt.legend(fontsize='large', ncol=ncol)
plt.xlim([-0.005, 0.15])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
secax.set_yticklabels(secax.get_yticklabels(), fontsize=14)
secax.set_ylabel(r'Resistivity ($\Omega$m)', fontsize=16, rotation=-90, labelpad=10)
# fig.savefig('E:/phd/NextCloud/Documents/ME_Transects/Swayze_paper/RoughFigures/EditFigures/feature_tests/archies-HS_host1000_zoom.png',
#             dpi=300)
plt.show()