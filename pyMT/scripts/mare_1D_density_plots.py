import pyMT.data_structures as DS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
plt.rcParams["figure.figsize"] = (16,10)


plot_titles = ['Rho_x', 'Rho_y', 'Rho_z', 'Response']
rho_bins = np.linspace(-0.5, 5.5, int(6/0.1) + 1)
# ani_bins = 20
ani_bins = np.linspace(-2.5, 2.5, int(5/0.1) + 1)
with PdfPages('E:/phd/NextCloud/VirtualBoxShare/2dAni/j2/1d/selected/renamed/all_models_density.pdf') as pdf:
	for site in ['1_MT/', '3_MT/','4_MT/', '5_MT/', '6_MT/', '7_MT/']:
		for hs in ['hs1000/', 'hs10000/']:
			for subdir in ['', 'highPrej/']:
				directory = 'E:/phd/NextCloud/VirtualBoxShare/2dAni/j2/1d/selected/renamed/' + site + hs + 'phase/' + subdir
				files = os.listdir(directory)
				files = [f for f in files if f.endswith('.csv')]
				plt.figure(figsize=[16, 10])
				models = []
				rho_x_hist = np.zeros((73, len(rho_bins) - 1))
				rho_y_hist = np.zeros((73, len(rho_bins) - 1))
				rho_z_hist = np.zeros((73, len(rho_bins) - 1))
				ani_ratio_hist = np.zeros((73, len(ani_bins) - 1))
				for f in files:
					iteration = f.split('.')[1]
					resp = DS.Data(directory + 'try10.{}.resp'.format(iteration), file_format='mare2dem')
					models.append(np.loadtxt(directory + f, delimiter=','))
					# plt.subplot(1,4,1)
					# plt.step(np.log10(model[:-5,1]), model[:-5,0] / 1000)
					# plt.subplot(1,4,2)
					# plt.step(np.log10(model[:-5,2]), model[:-5,0] / 1000)
					# plt.subplot(1,4,3)
					# plt.step(np.log10(model[:-5,3]), model[:-5,0] / 1000)
					# plt.subplot(1,4,4)
					# plt.semilogy(resp.sites[resp.site_names[0]].data['PhsZXY'], resp.periods, 'b-')
					# plt.semilogy(resp.sites[resp.site_names[0]].data['PhsZYX'], resp.periods, 'r-')
				# all_models = np.stack(model)
				if files:
					for ii in range(models[0].shape[0]):
						rho_x = [x[ii, 1] for x in models]
						rho_y = [x[ii, 2] for x in models]
						rho_z = [x[ii, 3] for x in models]
						ani_ratio = np.log10(np.array(rho_y) / np.array(rho_x))
						rho_x_hist[ii, :] = np.histogram(np.log10(rho_x), bins=rho_bins, weights=np.ones(len(rho_x))/len(rho_x))[0]
						rho_y_hist[ii, :] = np.histogram(np.log10(rho_y), bins=rho_bins, weights=np.ones(len(rho_x))/len(rho_x))[0]
						rho_z_hist[ii, :] = np.histogram(np.log10(rho_z), bins=rho_bins, weights=np.ones(len(rho_x))/len(rho_x))[0]
						ani_ratio_hist[ii, :] = np.histogram(ani_ratio, bins=ani_bins, weights=np.ones(len(rho_x))/len(rho_x))[0]

					plt.subplot(1,4,1)
					plt.pcolor(rho_bins, models[-1][:,0]/1000, rho_x_hist)
					plt.gca().set_xlabel('Log10 Rho')
					plt.gca().set_ylabel('Depth (km)')
					plt.gca().set_title('Rho x')
					plt.gca().set_ylim([0., 500])
					plt.gca().invert_yaxis()
					plt.subplot(1,4,2)
					plt.pcolor(rho_bins, models[-1][:,0]/1000, rho_y_hist)
					plt.gca().set_xlabel('Log10 Rho')
					plt.gca().set_ylabel('Depth (km)')
					plt.gca().set_title('Rho y')
					plt.gca().set_ylim([0., 500])
					plt.gca().invert_yaxis()
					plt.subplot(1,4,3)
					plt.pcolor(rho_bins, models[-1][:,0]/1000, rho_z_hist)
					plt.gca().set_xlabel('Log10 Rho')
					plt.gca().set_ylabel('Depth (km)')
					plt.gca().set_title('Rho z')
					plt.gca().set_ylim([0., 500])
					plt.gca().invert_yaxis()
					plt.subplot(1,4,4)
					plt.pcolor(ani_bins, models[-1][:,0]/1000, ani_ratio_hist)
					plt.gca().set_xlabel('Log10 y/x')
					plt.gca().set_ylabel('Depth (km)')
					plt.gca().set_title('Rho y / x')
					plt.gca().set_ylim([0., 500])
					plt.gca().invert_yaxis()
					plt.gcf().suptitle('{}, {} {}'.format(site[:-1], hs[:-1], subdir[:-1]))
					plt.gcf().tight_layout()
					pdf.savefig()
					plt.close()
				# try:
				# 	data = DS.Data(directory + 'MT_' + site[0] + '.emdata', file_format='mare2dem')
				# 	plt.semilogy(data.sites[data.site_names[0]].data['PhsZXY'], data.periods, 'bv')
				# 	plt.semilogy(data.sites[data.site_names[0]].data['PhsZYX'], data.periods, 'rv')
				# 	for ii in range(4):
				# 		plt.subplot(1,4,ii+1)
				# 		plt.gca().invert_yaxis()
				# 		plt.gca().set_title(plot_titles[ii])
				# 		if ii < 3:
				# 			plt.gca().set_xlim([0., 4.5])
				# 			plt.gca().set_xlabel('Log10 Rho')
				# 			plt.gca().set_ylabel('Depth (km)')
				# 		else:
				# 			plt.gca().set_xlabel('Log10 Period')
				# 			plt.gca().set_ylabel('Phase')
				# 	plt.gcf().suptitle('{}, {} {}'.format(site[:-1], hs[:-1], subdir[:-1]))
				# 	plt.gcf().tight_layout()
				# 	pdf.savefig()
				# 	plt.close()
				# except:
				# 	pass
				# 	print('Cant read files in {}'.format(directory))
				