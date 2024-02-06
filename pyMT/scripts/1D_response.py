import numpy as np
import matplotlib.pyplot as plt


MU = 4 * np.pi * 1e-7


def calculate_response(period_range, thickness, rho):
    scale = 1 / (4 * np.pi / 10000000)
    mu = 4 * np.pi * 1e-7
    periods = np.logspace(period_range[0], period_range[1], 80)
    omega = 2 * np.pi / periods
    # d = np.cumsum(.thickness + [100000])
    d = np.array(thickness) * 1000
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

    rhoa = 1/omega*np.abs(Z)**2;
    phi = np.angle(Z, deg=True);
    return rhoa, phi, Z, periods

thickness = [10, 20, 1000]
rho = [1, 100, 0.1]
period_range = [-2, 5]
rhoa1, phi1, Z1, periods = calculate_response(period_range, thickness, rho)
rho_data1 = Z1 * np.conj(Z1) * periods / (MU * 2 * np.pi)
pha_data1 = np.angle(Z1, deg=True)
rho = [10, 100, 0.1]
rhoa2, phi2, Z2, periods = calculate_response(period_range, thickness, rho)
rho_data2 = Z2 * np.conj(Z2) * periods / (MU * 2 * np.pi)
pha_data2 = np.angle(Z2, deg=True)
rho = [100, 100, 0.1]
rhoa3, phi3, Z3, periods = calculate_response(period_range, thickness, rho)
rho_data3 = Z3 * np.conj(Z3) * periods / (MU * 2 * np.pi)
pha_data3 = np.angle(Z3, deg=True)
plt.loglog(periods, rho_data1, 'r--', label='1 ohm-m')
plt.loglog(periods, rho_data2, 'g--', label='10 ohm-m')
plt.loglog(periods, rho_data3, 'b--', label='100 ohm-m')
plt.gca().set_xlabel('Period (s)')
plt.gca().set_ylabel('Apparent Resistivity (ohm-m)')
plt.legend()
plt.show()
plt.semilogx(periods, pha_data1, 'r--', label='1 ohm-m')
plt.semilogx(periods, pha_data2, 'g--', label='10 ohm-m')
plt.semilogx(periods, pha_data3, 'b--', label='100 ohm-m')
plt.gca().set_xlabel('Period (s)')
plt.gca().set_ylabel('Phase (degrees)')
plt.legend()
plt.show()
