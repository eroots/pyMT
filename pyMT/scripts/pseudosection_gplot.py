import pyMT.data_structures as DS
import matplotlib.pyplot as plt
import numpy as np
import pyMT.utils as utils
import pyMT.gplot as gplot
import os
from copy import deepcopy


local_path = 'E:'


out_path = local_path + '/phd/Nextcloud/Documents/ME_Transects/Upper_Abitibi/Paper/RoughFigures/feature_tests/'
file_exts = ['.png']
save_fig = 1
# base_path = 'E:/phd/NextCloud/data/Regions/MetalEarth/AG/Hex2Mod/MC-tests/'
base_path = 'E:/phd/NextCloud/data/Regions/MetalEarth/AG/Hex2Mod/UC-tests/10000ohm/'
list_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst'
data_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/AG/Hex2Mod/MC-tests/AG_depthTest_baseCase_resp.dat'
# resp_file = [base_path + 'UC{}/UC{}-10000ohm_resp.dat'.format(ii) for ii in range(1, 8)]
raw_data = DS.RawData(list_file)
data = DS.Data(data_file, listfile = list_file)
data.locations = raw_data.locations
cax = [[-10, 10], [-10, 10], [-0.3, 0.3]]

for ii in range(5, 8):
    tag = 'UC{}'.format(ii)
    out_path = local_path + '/phd/Nextcloud/Documents/ME_Transects/Upper_Abitibi/Paper/RoughFigures/feature_tests/{}/'.format(tag)
    resp_file = base_path + '{}/{}-10000ohm_resp.dat'.format(tag, tag)
    response = DS.Data(resp_file)
    fig = plt.figure(figsize=(24, 12))
    ax = fig.add_subplot(111)
    MV = gplot.MapView(fig=fig)
    MV.window['figure'] = fig
    MV.window['axes'] = [ax]
    # MV.colourmap = 'turbo'
    MV.site_data['data'] = data
    MV.site_data['response'] = response
    MV.site_names = data.site_names
    MV.padding_scale = 20
    MV.lut = 32
    MV.colourmap = 'bwr'
    MV.site_locations['all'] = data.locations
    MV.site_locations['generic'] = data.locations
    MV.markersize = 1
    for jj, period in enumerate(data.periods):
        for kk, fill_param in enumerate(['phaxy', 'phayx', 'tip']):
            MV.diff_cax = cax[kk]
            out_file = '{}_{}_p{}'.format(tag, fill_param, jj)
            MV.plan_pseudosection(data_type=['data', 'response'], fill_param=fill_param, n_interp=200, period_idx=jj)
            MV.plot_locations()
            MV.window['axes'][0].set_aspect(1)
            MV.window['axes'][0].set_title('Period = {:>6.2f} s'.format(period))

            if save_fig:
                for file_format in file_exts:
                    plt.savefig(out_path + out_file + file_format, dpi=300)
                    MV.window['axes'][0].clear()
                    # plt.close('all')
                    MV.window['colorbar'] = None
            else:
                plt.show()