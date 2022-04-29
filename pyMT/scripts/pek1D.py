import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess as sp
import os


######
# NOTE FOR FUTURE ERIC
# This script does not work as the Pek1D code requires libraries that aren't easy to install on windows
# The working version of this script is on your Dell laptop.

def plot_results(results):
	fig = plt.figure()
	axes = fig.add_subplot(1,2,1)
	# axes[0].loglog(results['period'], results['rhoxy'], 'bo')
	# axes[0].loglog(results['period'], results['rhoyx'], 'ro')
	axes.loglog(results['period'], results['phaxy'], 'bo')
	axes.loglog(results['period'], results['phayx'], 'ro')
	plt.show()


def create_input(model):
	with open(path+'an.dat', 'w') as f:
		f.write('{}\n'.format(model.shape[0]))
		for layer in range(model.shape[0]):
			f.write('{:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f}\n'.format(model[layer,0],
																					          model[layer,1],
																					          model[layer,2],
																					          model[layer,3],
																					          model[layer,4],
																					          model[layer,5],
																					          model[layer,6]))


path = 'E:/pek_santos_1D/1D_tests/'
nlayers = 5
model = np.zeros((nlayers,7))
model[:,0] = [10., 10, 10, 10, 0] # Layer thicknesses
model[:,1] = [1000, 300, 1, 1000, 20] # X Conductivities
model[:,2] = [1000, 300, 1000, 1000, 20] # Y Conductivities
model[:,3] = [1000, 300, 1000, 1000, 20] # Z Conductivities
model[:,4] = [0, 0, 0, 0, 0] # Anisotropic strike
model[:,5] = [0, 0, 0, 0, 0] # Anisotropic dip
model[:,6] = [0, 0, 0, 0, 0] # Anisotropic slant

create_input(model)

sp.run(path+'a.exe', stdout=sp.PIPE)
# os.system(path+'a.exe')

results = pd.read_csv(path+'anr.dat', header=None, names=['period',
														 'rhoxx', 'phaxx',
														 'rhoxy', 'phaxy', 
														 'rhoyx', 'phayx', 
														 'rhoyy', 'phayy'])

plot_results(results)