import pyMT.utils as utils
import pyMT.data_structures as WSDS
import matplotlib.pyplot as plt
import numpy as np
from pyMT import gplot

# listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\abi0_7\bigabi_2.lst'
# listfile2 = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\j2\center.lst'
# datafile1 = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\abi0_7\abi0_7.data'
# datafile2 = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\center0_sensTest\center_1.data'
# respfile1 = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\abi0_7\out_resp.00'
# respfile2 = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\center0_sensTest\sens_resp.01'
# datpath = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\j2'

# listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\Lalor\j2\lal0_redo2.lst'
# datafile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\Lalor\lal0_redo2\lal0_redo2.data'
# respfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\Lalor\lal0_redo2\fwd2_resp.00'

listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\wst\New\j2\cull6.lst'
datafile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\wst\New\wst0_5\wst0Inv_Final.data'
respfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\wst\New\wst0_5\wst0Final_resp.00'
dataset = WSDS.Dataset(listfile=listfile, responsefile=respfile, datafile=datafile)

fig = plt.figure(figsize=(26, 15))
# fig = Figure()
# fig.canvas = FigureCanvasAgg(fig)
dpm = gplot.DataPlotManager(fig=fig)
# dpm.fig = fig
# dpm.fig = fig
# snames = raw.site_names[:9]
# dpm.sites = ds1.get_sites(site_names=snames, dTypes='all')
# dpm.components = ['RHOXY']
# dpm.show_outliers = False
# dpm.plot_data()
# plt.show()

dpm.components = ('ZXYR', 'ZXYI', 'ZYXR', 'ZYXI')
# dpm.components = ('ZXXR', 'ZXXI', 'ZYYR', 'ZYYI')
# dpm.components = ['RhoXY', 'RhoYX', 'RhoXX', 'RhoYY']
dpm.markersize = 8
# for ii in range(0, len(dataset.data.site_names), 9):
for ii in range(24, 25):
    sites = dataset.data.site_names[ii:ii + 9]
    dpm.sites = dataset.get_sites(site_names=sites, dTypes='all')
    dpm.plot_data()
    dpm.fig.savefig(''.join(['PhaDiag_misfit', str(ii), '.pdf']), bbox_inches='tight', dpi=1000)
