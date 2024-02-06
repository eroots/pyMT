import pyMT.data_structures as DS
import numpy as np
import matplotlib.pyplot as plt

MU = 1/(4 * np.pi * 1e-7)
def calc_CART(Z, f):
	# Equation: p = (1j*MU/omega)*det(Z)*Z*(Z^-1).Transpose
	omega = f*2*np.pi
	T1 = (1j*MU/omega) * np.linalg.det(Z)
	p = T1 * np.matmul(Z, np.transpose(np.linalg.inv(Z)))
	return p

def calc_CART_mymath(Z, C, f):
	# Equation: p = (1j*MU/omega)*det(C)*det(Z)*C*Zr*(Zr^-1 * C^-1).Transpose
	# Note that MU here is actually 1/MU since the pyMT impedances are in different units than than the equations expect
	omega = f*2*np.pi
	T1 = (1j*MU/omega) * np.linalg.det(Z) * np.linalg.det(C)
	T2 = np.matmul(C, Z)
	T3 = np.transpose(np.matmul(np.linalg.inv(Z), np.linalg.inv(C)))
	p = T1 * np.matmul(T2, T3)
	return p


i_site = 8
data_R = DS.RawData('E:/phd/NextCloud/data/Regions/MetalEarth/CART_test/j2/test.lst')
data_D = DS.RawData('E:/phd/NextCloud/data/Regions/MetalEarth/CART_test/j2/distortion-staticOnly/test.lst')

site = data_R.site_names[i_site]
zxxr = data_R.sites[site].data['ZXXR']-1j*data_R.sites[site].data['ZXXI']
zxyr = data_R.sites[site].data['ZXYR']-1j*data_R.sites[site].data['ZXYI']
zyxr = data_R.sites[site].data['ZYXR']-1j*data_R.sites[site].data['ZYXI']
zyyr = data_R.sites[site].data['ZYYR']-1j*data_R.sites[site].data['ZYYI']
Z_R = np.array([[zxxr, zxyr], [zyxr, zyyr]])

zxxd = data_D.sites[site].data['ZXXR']-1j*data_D.sites[site].data['ZXXI']
zxyd = data_D.sites[site].data['ZXYR']-1j*data_D.sites[site].data['ZXYI']
zyxd = data_D.sites[site].data['ZYXR']-1j*data_D.sites[site].data['ZYXI']
zyyd = data_D.sites[site].data['ZYYR']-1j*data_D.sites[site].data['ZYYI']
Z_D = np.array([[zxxd, zxyd], [zyxd, zyyd]])


p_00 = np.zeros((data_R.sites[site].NP, 5), dtype=np.complex128)
p_01 = np.zeros((data_R.sites[site].NP, 5), dtype=np.complex128)
p_10 = np.zeros((data_R.sites[site].NP, 5), dtype=np.complex128)
p_11 = np.zeros((data_R.sites[site].NP, 5), dtype=np.complex128)
for ip in range(data_R.sites[site].NP):
	C = np.real(np.matmul(Z_D[:,:,ip], np.linalg.inv(Z_R[:,:,ip]))) # Calculate the distortion tensor

	# p_a of Z-regional (re-calculated from data)
	p_R1 = np.flip(calc_CART(Z_R[:,:,ip], 1/data_R.sites[site].periods[ip]))
	# p_a of Z-distorted (re-calculated from distorted data)
	p_D1 = np.flip(calc_CART(Z_D[:,:,ip], 1/data_D.sites[site].periods[ip]))
	# p_a of Z-regional (from pyMT data)
	p_R2 = data_R.sites[site].CART[ip].Ua + 1j*data_R.sites[site].CART[ip].Va
	# p_a of Z-distorted (from pyMT data)
	p_D2 = data_D.sites[site].CART[ip].Ua + 1j*data_D.sites[site].CART[ip].Va
	# Z-distorted by applying C to Z-regional
	Z_D2 = np.matmul(C,Z_R[:,:,ip])
	# p_a of Z-distorted (from C*data_regional)
	p_D3 = np.flip(calc_CART(np.matmul(C, Z_R[:,:,ip]), 1/data_R.sites[site].periods[ip])) 
	# Distorted p_a according to the paper (p_a=C*p_aR)
	p_claim = np.matmul(C, p_R1) 
	# p_a according to the math I worked out for how C should factor into p_a
	p_mymath = np.flip(calc_CART_mymath(Z_R[:,:,ip], C, 1/data_R.sites[site].periods[ip]))

	p_00[ip,0] = p_D1[0,0] # p_a of Z-distorted (re-calculated from distorted data)
	p_00[ip,1] = p_D2[0,0] # p_a of Z-distorted (directly from pyMT data)
	p_00[ip,2] = p_D3[0,0] # p_a of Z-distorted (from C*data_regional)
	p_00[ip,3] = p_claim[0,0] # Distorted p_a according to the paper (p_a=C*p_aR)

	p_01[ip,0] = p_D1[0,1]
	p_01[ip,1] = p_D2[0,1]
	p_01[ip,2] = p_D3[0,1]
	p_01[ip,3] = p_claim[0,1]

	p_10[ip,0] = p_D1[1,0]
	p_10[ip,1] = p_D2[1,0]
	p_10[ip,2] = p_D3[1,0]
	p_10[ip,3] = p_claim[1,0]

	p_11[ip,0] = p_D1[1,1]
	p_11[ip,1] = p_D2[1,1]
	p_11[ip,2] = p_D3[1,1]
	p_11[ip,3] = p_claim[1,1]

	p_00[ip,4] = p_mymath[0,0] # p_a according to the math I worked out for how C should factor into p_a
	p_01[ip,4] = p_mymath[0,1]
	p_10[ip,4] = p_mymath[1,0]
	p_11[ip,4] = p_mymath[1,1]

markers = ['k-', 'r--', 'bx', 'g.', 'mv']
# markers = ['']
plt.figure()
for ii, v in enumerate([p_00, p_01, p_10, p_11]):
	plt.subplot(2,2,ii+1)
	p_real = np.real(v)
	p_imag = np.imag(v)
	for jj, calc in enumerate((r'$pyMT p_{a}(Z_{D})$', r'$pyMT p_{a}=U_{a}+iV_{a}$', r'Auto-sub $p_{a}(CZ^{r})$', r'Hering $p_{a}=C*p_{a}(Z^{r})$', 'Manual sub')):
		# for jj, calc in enumerate(['D1', 'D2', 'D3']):
		plt.semilogx(data_R.sites[site].periods, p_real[:, jj], markers[jj], label=calc)
		plt.xlabel('Period (s)')
		# plt.semilogx(data_R.sites[site].periods, 100*(p_real[:, 4]-p_real[:,2])/p_real[:, 2], markers[jj], label=calc)
		# plt.title('Real, index {}'.format(ii))
		plt.semilogx(data_R.sites[site].periods, p_imag[:, jj], markers[jj])
		# plt.semilogx(data_R.sites[site].periods, 100*(p_imag[:, 4]-p_imag[:,2])/p_imag[:, 2], markers[jj])
		# plt.title('Imag, index {}'.format(ii))
plt.legend()
plt.show()
	# print('p_R1-p_R2')
	# print(p_R1 - p_R2)
	# print('p_D1-p_D2')
	# print(p_D1 - p_D2)
	# print('Z_D-C*Z_R')
	# print(Z_D[:,:,ip]-Z_D2)
	# print('p(CZ_R)-p_D2')
	# print(p_D3-p_D1)



