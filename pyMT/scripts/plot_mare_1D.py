import pyMT.data_structures as DS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
plt.rcParams["figure.figsize"] = (16,10)


plot_titles = ['Rho_x', 'Rho_y', 'Rho_z', 'Response']
with PdfPages('E:/phd/NextCloud/VirtualBoxShare/2dAni/j2/1d/selected/renamed/all_models.pdf') as pdf:
	for site in ['1_MT/', '3_MT/','4_MT/', '5_MT/', '6_MT/', '7_MT/']:
		for hs in ['hs1000/', 'hs10000/']:
			for subdir in ['', 'highPrej/']:
				directory = 'E:/phd/NextCloud/VirtualBoxShare/2dAni/j2/1d/selected/renamed/' + site + hs + 'phase/' + subdir
				files = os.listdir(directory)
				files = [f for f in files if f.endswith('.csv')]
				plt.figure(figsize=[16, 10])
				for f in files:
					iteration = f.split('.')[1]
					resp = DS.Data(directory + 'try10.{}.resp'.format(iteration), file_format='mare2dem')
					model = np.loadtxt(directory + f, delimiter=',')
					plt.subplot(1,4,1)
					plt.step(np.log10(model[:-5,1]), model[:-5,0] / 1000)
					plt.subplot(1,4,2)
					plt.step(np.log10(model[:-5,2]), model[:-5,0] / 1000)
					plt.subplot(1,4,3)
					plt.step(np.log10(model[:-5,3]), model[:-5,0] / 1000)
					plt.subplot(1,4,4)
					plt.semilogy(resp.sites[resp.site_names[0]].data['PhsZXY'], resp.periods, 'b-')
					plt.semilogy(resp.sites[resp.site_names[0]].data['PhsZYX'], resp.periods, 'r-')
				try:
					data = DS.Data(directory + 'MT_' + site[0] + '.emdata', file_format='mare2dem')
					plt.semilogy(data.sites[data.site_names[0]].data['PhsZXY'], data.periods, 'bv')
					plt.semilogy(data.sites[data.site_names[0]].data['PhsZYX'], data.periods, 'rv')
					for ii in range(4):
						plt.subplot(1,4,ii+1)
						plt.gca().invert_yaxis()
						plt.gca().set_title(plot_titles[ii])
						if ii < 3:
							plt.gca().set_xlim([0., 4.5])
							plt.gca().set_xlabel('Log10 Rho')
							plt.gca().set_ylabel('Depth (km)')
						else:
							plt.gca().set_xlabel('Log10 Period')
							plt.gca().set_ylabel('Phase')
					plt.gcf().suptitle('{}, {} {}'.format(site[:-1], hs[:-1], subdir[:-1]))
					plt.gcf().tight_layout()
					pdf.savefig()
					plt.close()
				except:
					pass
				# 	print('Cant read files in {}'.format(directory))
				