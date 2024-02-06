import pyMT.data_structures as WSDS
import pyMT.utils as utils
import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import griddata
import pandas as pd
import segyio
# from scipy.interpolate import SmoothBivariateSpline as RBS
import copy
from shutil import copyfile



#local_path = 'C:/Users/eroots'
#local_path = 'C:/Users/eric/'
local_path = 'E:'
# local_path = 'E:/'



def interpolate_slice(x, y, Z, NP):
    mod_interp = RBS(x, y, Z)
    interp_vals = mod_interp(np.linspace(x[0], x[-1], NP),
                             np.linspace(y[0], y[-1], NP))
    return interp_vals


def project_locations(data, zone, letter):
    data.locations = data.get_locs(mode='latlong')
    for ii in range(len(data.locations)):
        easting, northing = utils.project((data.locations[ii, 1],
                                           data.locations[ii, 0]),
                                          zone=zone, letter=letter)[2:]
        data.locations[ii, 1], data.locations[ii, 0] = easting, northing
    return data

# seisline = ['12','14','15','16','16a','23','25','28']
seisline = ['21']
# seisline += ['17','18','21','24','27']
# seismic = pd.read_table(local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/rewesternsuperiorandabitibimt/{}.cdp'.format(seisline),
#                         # header=0, names=('trace', 'x', 'y'),  sep='\s+')
#                         header=0, names=('ffid', 'trace', 'x', 'y', 'z'), sep='\s+')
depth_conversion_velocity = 6.500  # in km/s
project_model = 1
offset_model = 1
zlim = [0, 50]
dz = 0.026  # in km
times = np.linspace(0, 15000, 15001)
vel_model = np.hstack([np.linspace(6000, 6350, 1000),
                       np.linspace(6350, 6500, 3000),
                       np.linspace(6500, 6700, 4000),
                       np.linspace(6850, 7250, 5000),
                       np.linspace(8250, 8400, 2001)])
for line in seisline:
    projected_model_segy =  'D:/OpenDTect_data/RawData/projcoord/line{}_velocity.sgy'.format(line, line)
    seismic_data_path = local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/segy/feat_line{}_segy/line{}_curvelet.sgy'.format(line, line)

    if offset_model:
        model_offset = 500
    # segyio.tools.from_array(smooth_model_segy, vals, dt=dz)
    with segyio.open(seismic_data_path, 'r', strict=False) as src:
        qx = [t[segyio.TraceField.GroupX] / 10000 for t in src.header]
        qy = [t[segyio.TraceField.GroupY] / 10000 for t in src.header]
        qx_project = np.linspace(qx[0], qx[-1], len(qx))
        qy_project = np.linspace(qy[0], qy[-1], len(qy))
        xdiff = abs(qx_project[0] - qx_project[-1])
        ydiff = abs(qy_project[0] - qy_project[-1])
        if xdiff > ydiff:
            x_offset = 500
            y_offset = 0
        else:
            y_offset = 500
            x_offset = 0
        # dt_new = int(round(2 * 1000 * 1000 * dz / depth_conversion_velocity))
        spec=segyio.tools.metadata(src)
        # spec.samples = np.arange(len(qz)) * dt_new / 1000
        with segyio.create(projected_model_segy, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            dst.header = src.header
            dst.trace = src.trace
            step = dst.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL] / 1000
            trace_times = np.linspace(0, step * len(dst.trace[0]), len(dst.trace[0]))
            vels = np.interp(trace_times, times, vel_model)
            c = 0
            for ix, trace in enumerate(dst.trace):
                dst.trace[ix] = vels
            for ix, header in enumerate(dst.header):
                # header.update({segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_new})
                if ix < len(qx):
                    header.update({segyio.TraceField.GroupX: int(qx_project[ix] * 10000 + x_offset)})
                    header.update({segyio.TraceField.GroupY: int(qy_project[ix] * 10000 + y_offset)})
                    header.update({segyio.TraceField.SourceX: int(qx_project[ix] * 10000 + x_offset)})
                    header.update({segyio.TraceField.SourceY: int(qy_project[ix] * 10000 + y_offset)})
                    header.update({segyio.TraceField.CDP_X: int(qx_project[ix] * 10000 + x_offset)})
                    header.update({segyio.TraceField.CDP_Y: int(qy_project[ix] * 10000 + y_offset)})
                else:
                    header.update({segyio.TraceField.GroupX: int(qx_project[-1] * 10000 + x_offset + c)})
                    header.update({segyio.TraceField.GroupY: int(qy_project[-1] * 10000 + y_offset + c)})
                    header.update({segyio.TraceField.SourceX: int(qx_project[-1] * 10000 + x_offset + c)})
                    header.update({segyio.TraceField.SourceY: int(qy_project[-1] * 10000 + y_offset + c)})
                    header.update({segyio.TraceField.CDP_X: int(qx_project[-1] * 10000 + x_offset + c)})
                    header.update({segyio.TraceField.CDP_Y: int(qy_project[-1] * 10000 + y_offset + c)})
                    c += 50