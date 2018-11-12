import pyMT.utils as utils
import pyMT.data_structures as WSDS
import matplotlib.pyplot as plt
import numpy as np
from pyMT import gplot
from matplotlib.backends.backend_pdf import PdfPages


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

# listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\wst\New\j2\cull6.lst'
# datafile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\wst\New\wst0_5\wst0Inv_Final.data'
# respfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\wst\New\wst0_5\wst0Final_resp.00'

listfile = r'C:\Users\eric\phd\ownCloud\data\Regions\MetalEarth\dryden\j2\dry5_3.lst'
datafile = r'C:\Users\eric\phd\ownCloud\data\Regions\MetalEarth\dryden\dry5\dry5_3.dat'
respfile = r'C:\Users\eric\phd\ownCloud\data\Regions\MetalEarth\dryden\dry5\dry5_3_lastIter.dat'
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
components = {'diag': ('ZXXR', 'ZXXI', 'ZYYR', 'ZYYI'),
              'offdiag': ('ZXYR', 'ZXYI', 'ZYXR', 'ZYXI'),
              'phase': ('PhaXY', 'PhaYX', 'PhaXX', 'PhaYY'),
              'rho': ('RhoXY', 'RhoYX', 'RhoXX', 'RhoYY'),
              'tipper': ('TZXR', 'TZXI', 'TZYR', 'TZYI')}
# dpm.components = ('ZXYR', 'ZXYI', 'ZYXR', 'ZYXI')
# dpm.components = ('ZXXR', 'ZXXI', 'ZYYR', 'ZYYI')
out_path = r'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/dryden/'
dpm.components = ['RhoXY', 'RhoYX', 'RhoXX', 'RhoYY']
dpm.show_outliers = False
dpm.markersize = 8
dpm.outlier_thresh = 3
dpm.wrap = True
sites_per_page = 9
with PdfPages(''.join([out_path, 'Dryden-pha.pdf'])) as pdf:
    for comps in ['rho']:
        # for comps in ['tipper']:
        # dpm.components = components[comps]
        dpm.components = ['PhaXY', 'PhaYX']
        for ii in range(0, len(dataset.data.site_names), sites_per_page):
            if comps == 'tipper':
                dpm.scale = 'none'
            else:
                dpm.scale = 'sqrt(periods)'
            sites = dataset.data.site_names[ii:ii + sites_per_page]
            dpm.sites = dataset.get_sites(site_names=sites, dTypes='all')
            dpm.plot_data()
            # dpm.fig.savefig(''.join([comps, '_misfit', str(ii), '.pdf']), bbox_inches='tight', dpi=1000)
            pdf.savefig()
            # plt.close()
