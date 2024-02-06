## Calculate approximate anisotropic resistivities from a slice through an isotropic model
# Uses equations from Eisel and Haak (1999)
import pyMT.data_structures as DS
import pyMT.utils as utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

model = DS.Model('E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wstZK_lastIter.rho')


rho_cutoff_low = 10**2.5
rho_cutoff_high = 10**3
# Take a line through the 84th X, 184500 m
stack = np.mean(np.log10(model.vals[84,:,55:61]), axis=1) # Keep stack in log10 space
X = utils.edge2center(model.dy)
interpolator = interp1d(X, stack, kind='linear')
# Interpolate assuming values and linearly transisition (in log10 space)
X_interp = np.linspace(X[0], X[-1], int((X[-1] - X[0]) / 2.5))
stack_interp = interpolator(X_interp)

idx_low = np.where(np.diff(np.sign(stack_interp - np.log10(rho_cutoff_low))))[0]
idx_high = np.where(np.diff(np.sign(stack_interp - np.log10(rho_cutoff_high))))[0]

plt.semilogy(X_interp, stack_interp)
plt.semilogy(X_interp[idx_low], stack_interp[idx_low], 'x')
plt.semilogy(X_interp[idx_high], stack_interp[idx_high], 'o')
plt.show()
# For high resistivity zones
lowRho_width = []
highRho_width = []
lowRho_logAvg = []
highRho_logAvg = []
print('Resistive zones:')
for ii in range(1,5):
    highRho_width.append(abs(X_interp[idx_high[(ii*2)-1]]-X_interp[idx_high[(ii*2)]]))
    highRho_logAvg.append(10**np.mean(stack_interp[idx_high[ii*2-1]:idx_high[(ii*2)]]))
    print('Width of zone {} is: {:<10.0f}'.format(ii,abs(X_interp[idx_high[(ii*2)-1]]-X_interp[idx_high[(ii*2)]])))
    print('Mean rho is: {:<10.0f}'.format(np.mean(10**stack_interp[idx_high[ii*2-1]:idx_high[(ii*2)]])))
    print('LogMean rho is: {:<10.0f}'.format(10**np.mean(stack_interp[idx_high[ii*2-1]:idx_high[(ii*2)]])))

# For low resistivity zones
print ('\nConductive Zones:')
for ii in range(1,4):
    lowRho_width.append(abs(X_interp[idx_low[(ii*2)]]-X_interp[idx_low[(ii*2+1)]]))
    lowRho_logAvg.append(10**np.mean(stack_interp[idx_low[ii*2]:idx_low[(ii*2+1)]]))
    print('Width of zone {} is: {:<10.0f}'.format(ii,abs(X_interp[idx_low[(ii*2)]]-X_interp[idx_low[(ii*2+1)]])))
    print('Mean rho is: {:<10.0f}'.format(np.mean(10**stack_interp[idx_low[ii*2]:idx_low[(ii*2+1)]])))
    print('LogMean rho is: {:<10.0f}'.format(10**np.mean(stack_interp[idx_low[ii*2]:idx_low[(ii*2+1)]])))

avgRho_high = 10**np.mean(np.log10(highRho_logAvg))
avgRho_low  = 10**np.mean(np.log10(lowRho_logAvg))
avgWidth_high = np.mean(highRho_width)
avgWidth_low = np.mean(lowRho_width)

p1, p2 = avgRho_low, avgRho_high
d1, d2 = avgWidth_low, avgWidth_high

rho_para = (p1*p2*(d1+d2)) / (p1*d2 + p2*d1)
rho_perp = (p1*d2 + p2*d2) / (d1 + d2)

print('Rho Perpendicular: {:10.0f}\n Rho Parallel: {:10.0f}'.format(rho_perp, rho_para))