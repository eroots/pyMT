import pyMT.utils as utils
import pyMT.data_structures as WSDS
import matplotlib.pyplot as plt
import numpy as np
from pyMT import gplot


listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\abi0_7\bigabi_2.lst'
listfile2 = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\j2\center.lst'
datafile1 = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\abi0_7\abi0_7.data'
datafile2 = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\center0_sensTest\center_1.data'
respfile1 = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\abi0_7\out_resp.00'
respfile2 = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\center0_sensTest\sens_resp.01'
datpath = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\New\j2'

ds1 = WSDS.Dataset(listfile=listfile, datafile=datafile1,
                   responsefile=respfile1, datpath=datpath)
ds2 = WSDS.Dataset(listfile=listfile2, datafile=datafile2,
                   responsefile=respfile2, datpath=datpath)
raw = WSDS.RawData(listfile=listfile, datpath=datpath)
fig = plt.figure()
dpm = gplot.DataPlotManager(fig=fig)
snames = raw.site_names[:9]
dpm.sites = ds1.get_sites(site_names=snames, dTypes='all')
dpm.components = ['RHOXY']
dpm.show_outliers = False
dpm.plot_data()
plt.show()


