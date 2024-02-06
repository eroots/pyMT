import pandas as pd
from pykrige.ok import OrdinaryKriging as ok
# from pykrige.uk import UniversalKriging as uk
# from skgstat import OrdinaryKriging
# from skgstat import Variogram
import numpy as np
import pyMT.data_structures as DS
import matplotlib.pyplot as plt
from pyMT.e_colours import colourmaps as cm
import pyMT.utils as utils


# Remind of terms:
# Variogram fits a model to the autocorrelation of the data points. This model is what is used to determine the interpolation. The model uses the following parameters:
# Nugget is the y-intercept of the variogram. A nugget of 0 basically means data at lag-0 (colocated data) are perfect. Non-zero nugget should be used for imperfect data (always?)
# Sill defines at what y-axis point the fitted curve levels off (theoretically after a certain distance your data should all have equal autocorrelation?)
# Range defines the point along the x-axis that the sill is reached.

data = DS.RawData('E:/phd/NextCloud/data/regions/MetalEarth/j2/upper_abitibi_hex.lst')
out_path = 'E:/phd/NextCloud/Documents/ME_Transects/Upper_Abitibi/Paper/RoughFigures/moho_depth/exp/'
file_exts = ['.png']
dpi = 300
save_fig = 0
internal_padding = 3
external_padding = 8
# internal_padding = 100000
# external_padding = 800000
interval = .03
mohodata = pd.read_csv('C:/Users/user/Downloads/of_8243/of_8243.csv', header=0, sep=',', encoding='ISO-8859-1')
idx = np.isnan(mohodata['STDDEV'])
medstd = np.mean(mohodata['STDDEV'][~idx])
mohodata['STDDEV'][idx] = -999
mohodata = mohodata[mohodata['STDDEV'] < 5]
idx = mohodata['STDDEV'] == -999
mohodata['STDDEV'][idx] = np.nan
# mohodata[idx] = np.nan

data.locations = data.get_locs(mode='latlong')
# data.locations[:, 1], data.locations[:, 0] = utils.to_lambert(data.locations[:,0], data.locations[:,1])
# 
# mohodata['LON'], mohodata['LAT'] = utils.to_lambert(mohodata['LAT'], mohodata['LON'])

minlat = min(data.locations[:,0])
maxlat = max(data.locations[:,0])
minlon = min(data.locations[:,1])
maxlon = max(data.locations[:,1])

gridx = np.arange(minlon-internal_padding, maxlon+internal_padding + interval, interval)
gridy = np.arange(minlat-internal_padding, maxlat+internal_padding + interval, interval)

# data_include = mohodata[(mohodata['LAT'] >minlat-external_padding) * (mohodata['LAT'] < maxlat+external_padding) *
#                         (mohodata['LON'] > minlon - external_padding) * (mohodata['LON'] < maxlon+external_padding)]

data_include = mohodata[(mohodata['LAT'] > 43) * (mohodata['LAT'] < 60) *
                        (mohodata['LON'] > -100) * (mohodata['LON'] < -70)]

# data_include['LON'], data_include['LAT'] = utils.to_lambert(data_include['LAT'], data_include['LON'])
# for ii in range(len(data_include)):
#     data_include['LON'].iloc[ii], data_include['LAT'].iloc[ii] = utils.project(np.array((data_include['LON'].iloc[ii],
#                                                                                          data_include['LAT'].iloc[ii])).T,
#                                                                                          zone=17, letter='U')[2:]
    
roi = data_include[(data_include['LAT'] >minlat-internal_padding) * (data_include['LAT'] < maxlat+internal_padding) *
               (data_include['LON'] > minlon - internal_padding) * (data_include['LON'] < maxlon+internal_padding)]




# for model in ['linear']:
    # for weighting in [True]:
nlags = 24
sills = [70]
rranges = [30]
nuggets = [1]
method = 'ordinary'
weighting = True
# for method in ['ordinary']:#, 'universal']:
for model in ['exponential']:#, 'spherical']:
    # for slope in [0.25, 0.5, 1, 2, 4, 8]:
    for nugget in nuggets:
        # for slope in [2]:
            # for nugget in [0.5]:
        for rrange in rranges:
            for sill in sills:
                # sill = sillmult * nugget
            # for nlags in (6, 12, 24, 48, 96, 192):
                out_file = 'mohoallWGS_depth_model-{}_sill-{}_range-{}_nugget-{}'.format(model, sill, rrange, nugget)
                # out_file = 'mohoallWGS_depth_model-{}_slope-{}_nugget-{}'.format(model, slope, nugget)
                if method == 'ordinary':
                    # V = Variogram(coordinates=np.array((data_include['LON'], data_include['LAT'])).T,
                    #               values=data_include['MOHODEPTH'], n_lags=20, estimator='dowd',
                    #               model='stable', fit_sigma='linear', use_nugget=True)
                    # krig = OrdinaryKriging(V, min_points=5, max_points=20, mode='exact')
                    krig = ok(data_include['LON'], data_include['LAT'], data_include['MOHODEPTH'],
                              variogram_model=model,
                              coordinates_type='geographic',
                              exact_values=False,
                              pseudo_inv=True,
                              weight=weighting,
                              nlags=nlags,
                              variogram_parameters={'nugget': nugget, 'sill': sill, 'range': rrange})
                              # variogram_parameters={'nugget': nugget})
                elif method == 'universal':
                    krig = uk(data_include['LON'], data_include['LAT'], data_include['MOHODEPTH'],
                              variogram_model=model,
                              exact_values=False,
                              pseudo_inv=True,
                              weight=weighting,
                              nlags=nlags)
                # X, Y = np.meshgrid(gridx, gridy)
                # z = krig.transform(X.flatten(), Y.flatten()).reshape(X.shape)
                z, ss = krig.execute('grid', gridx, gridy)
                plt.figure(figsize=(20, 12))
                plt.pcolor(gridx, gridy, z, cmap=cm.get_cmap('bgy_r', 32), vmin=30, vmax=50, zorder=0)
                plt.plot(data.locations[:,1], data.locations[:,0], 'k.', markersize=5, zorder=3)
                plt.scatter(roi['LON'],roi['LAT'], c=roi['MOHODEPTH'],
                            cmap=cm.get_cmap('bgy_r', 32), vmin=30, vmax=50, zorder=1,
                            s=250)# s=1/np.sqrt(roi['STDDEV'])*20)
                cb1 = plt.colorbar()
                cb1.set_label('Moho Depth (km)', rotation=270, labelpad=30, fontsize=14)
                plt.scatter(roi['LON'],roi['LAT'], c=roi['STDDEV'],
                            edgecolor='k', cmap=cm.get_cmap('turbo', 32), zorder=2,
                            s=50)# s=1/np.sqrt(roi['STDDEV'])*20)
                cb2 = plt.colorbar()
                cb2.set_label('Uncertainty (km)', rotation=270, labelpad=30, fontsize=14)
                cs = plt.contour(gridx, gridy, z, np.arange(25, 55, 2.5), colors='k')
                plt.gca().clabel(cs, inline=True, fontsize=10, fmt='%2.1f', colors='k', inline_spacing=2.5)
                # plt.gca().clabel(cs, inline=True, fontsize=10, fmt='%2.2f', colors='k', inline_spacing=5)
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.gca().set_xlim([minlon-internal_padding, maxlon+internal_padding])
                plt.gca().set_ylim([minlat-internal_padding, maxlat+internal_padding])
                plt.gca().set_aspect('equal')
                if save_fig:
                    for file_format in file_exts:
                        # manager = plt.get_current_fig_manager()
                        # manager.window.showMaximized()
                        plt.savefig(out_path + out_file + file_format, dpi=dpi,
                                    transparent=True)
                    plt.close('all')
                else:
                    plt.show()

