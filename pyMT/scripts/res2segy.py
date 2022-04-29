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

# UPPER-ABITIBI
main_transect = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/ROUBB.lst')
data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst')
backup_data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst')
mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/AG/Hex2Mod/HexAG_Z_static.model')
# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/Hex2Mod/HexAG-test300ohm_block.model')
# seismic = pd.read_table(local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/rewesternsuperiorandabitibimt/12.cdp',
#                         header=0, names=('trace', 'x', 'y'), sep='\s+')
# seismic_data_path = local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/segy/feat_line12_segy/line12_curvelet.sgy'
# seismic = pd.read_table(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/AG/seismic/cdp_utm.dat',
#                         header=0, names=('trace', 'x', 'y', 'z', 'dummy'), sep='\s+')
# seismic_data_path = local_path + '/phd/Nextcloud/data/Regions/MetalEarth/AG/seismic/merge_lmig_curvelet.sgy'
# seisline = ['12','14','15','16','16a','23','25','28']
seisline = ['21']
# seisline += ['17','18','21','24','27']

# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/Hex2Mod/HexAG_Z_only.model')
# main_transect = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/ROUBB.lst')
# data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst')
# backup_data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst')
# # mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/Hex2Mod/HexAG_Z_only.model')
# # mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/Hex2Mod/HexAG-test300ohm_block.model')
# seismic = pd.read_table(local_path + '/phd/Nextcloud/Metal Earth/Data/Seismic/ME_Seismic_PostStack_Migrated_sections/' +
#                         r'ROUYN_LN141_R1_KMIG/ROUYN_LN141_R1_KMIG_SUGETHW_UTM.txt',
#                         header=0, names=('trace', 'x', 'y'), sep='\s+')
# seismic_lines = ['17','18','21','24','27']
# seismic_lines = ['14','15','16','16a','25','28']
# seisline = '27'
# seismic = pd.read_table(local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/rewesternsuperiorandabitibimt/{}.cdp'.format(seisline),
#                         # header=0, names=('trace', 'x', 'y'),  sep='\s+')
#                         header=0, names=('ffid', 'trace', 'x', 'y', 'z'), sep='\s+')
depth_conversion_velocity = 6.500  # in km/s
project_model = 1
project_segy = 0
smooth_model = 0
smooth_segy = 0
offset_model = 1
zlim = [0, 50]
dz = 0.026  # in km
all_backups = {'model': copy.deepcopy(mod), 'main_transect': copy.deepcopy(main_transect),
               'data': copy.deepcopy(data), 'backup_data': copy.deepcopy(backup_data)}
for line in seisline:
    smooth_segy_file = 'D:/OpenDTect_data/RawData/smooth/line{}_smooth.sgy'.format(line, line)
    projected_segy_file = 'D:/OpenDTect_data/RawData/segy_coord_exact/line{}.sgy'.format(line, line)
    smooth_model_segy =  'D:/OpenDTect_data/RawData/smooth/line{}_resistivity.sgy'.format(line, line)
    projected_model_segy =  'D:/OpenDTect_data/RawData/projcoord/line{}_resistivity_offset.sgy'.format(line, line)
    file_path = local_path + '/phd/NextCloud/Documents/ME_Transects/Upper_Abitibi/Paper/RoughFigures/alongSeis/line{}/'.format(line)
    # if line in ['12', '14']:
    #     seismic = pd.read_table(local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/rewesternsuperiorandabitibimt/{}.cdp'.format(line),
    #                         header=0, names=('trace', 'x', 'y'), sep='\s+')
    # else:
    #     seismic = pd.read_table(local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/rewesternsuperiorandabitibimt/{}.cdp'.format(line),
    #                     header=0, names=('trace', 'dummy', 'x', 'y', 'z'), sep='\s+')
    seismic_data_path = local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/segy/feat_line{}_segy/line{}_curvelet.sgy'.format(line, line)

    with segyio.open(seismic_data_path, 'r', strict=False) as src:
         # qx = [t[segyio.TraceField.GroupX] / 10000 for t in src.header]
         sx = [t[segyio.TraceField.SourceX] / 10000 for t in src.header]
         # qy = [t[segyio.TraceField.GroupY] / 10000 for t in src.header]
         sy = [t[segyio.TraceField.SourceY] / 10000 for t in src.header]
         qx_project = [t[segyio.TraceField.GroupX] / 10000 for t in src.header]
         qy_project = [t[segyio.TraceField.GroupY] / 10000 for t in src.header]

    # qx_project = np.linspace(qx[0], qx[-1], len(qx))
    # qy_project = np.linspace(qy[0], qy[-1], len(qy))
    qx, qy = copy.deepcopy(qx_project), copy.deepcopy(qy_project)
    mod = copy.deepcopy(all_backups['model'])
    data = copy.deepcopy(all_backups['data'])
    main_transect = copy.deepcopy(all_backups['main_transect'])
    backup_data = copy.deepcopy(all_backups['backup_data'])


    data = copy.deepcopy(main_transect)
    origin = backup_data.origin  # Has to be this since 'data' has been modified and so is no longer in sync with model
    # mod.origin = origin
    mod.origin = origin
    mod.to_UTM()
    if mod.coord_system == 'UTM':
        mod.dx = [xx / 1000 for xx in mod.dx]
        mod.dy = [yy / 1000 for yy in mod.dy]
        mod.dz = [zz / 1000 for zz in mod.dz]


    # qx, qy = (np.array(seismic['x'] / 1000),
    #           np.array(seismic['y']) / 1000)

    qz = np.linspace(0, zlim[1], int(round(zlim[1] / dz)))

    vals = np.log10(mod.vals)

    #  Build Mx3 array of data points
    data_points = np.zeros((mod.vals.size, 3))
    data_values = np.zeros((mod.vals.size))
    print('Number of data points: {}'.format(data_values.size))
    cc = 0
    x = utils.edge2center(mod.dx)
    y = utils.edge2center(mod.dy)
    z = utils.edge2center(mod.dz)
    for ix in range(len(x)):
        for iy in range(len(y)):
            for iz in range(len(z)):
                data_values[cc] = vals[ix, iy, iz]
                data_points[cc, :] = np.array((y[iy], x[ix], z[iz]))
                cc += 1

    query_points = np.zeros((len(qx) * len(qz), 3))
    #  Build Nx3 array of query points
    cc = 0
    print('Number of query points: {}'.format(query_points.size))
    for ix in range(len(qx)):
            for iz in qz:
                query_points[cc, :] = np.array((qx[ix], qy[ix], iz))
                cc += 1


    # vals = griddata(data_points, data_values, query_points, 'nearest')
    interpolator = RGI((y, x, z), np.transpose(vals, [1, 0, 2]), bounds_error=False, fill_value=5)
    vals = interpolator(query_points)
    vals = np.reshape(vals, [len(qx), len(qz)])
    
    
    # with segyio.open(seismic_data_path, strict=False) as src:
    #     seis_data = np.stack([t.astype(np.float) for t in src.trace])
    #     x = np.array([t[segyio.TraceField.GroupX] for t in src.header])
    #     y = np.array([t[segyio.TraceField.GroupY] for t in src.header])
    #     cdp = np.array([t[segyio.TraceField.CDP] for t in src.header])
    #     header = src.text[0].decode('ascii')
    #     dt = src.bin[segyio.BinField.Interval] / 1000


    # if smooth_segy:
    #     copyfile(seismic_data_path, smooth_segy_file)
    #     # with segyio.open()
    #     with segyio.open(smooth_segy_file, 'r+', strict=False, ignore_geometry=True) as spec:
    #         # spec = segyio.tools.metadata(src)
    #         t2d = spec.samples
    #         spec._samples = spec.samples * depth_conversion_velocity / 2
    #         cutoff = np.argmin(np.abs(spec.samples - zlim[1]))
    #         spec._samples = spec.samples[:cutoff]
    # if model_to_segy:
    #     copyfile(seismic_data_path, smooth_model_segy)
    #     with segyio.open(smooth_model_segy, 'r+', strict=False, ignore_geometry=True) as spec:
    #         spec._samples = qz
    #         for ix, trace in enumerate(spec.trace):
    #             spec.trace[ix] = vals[ix, :]
    #         spec.bin.update(hdt=dz)
    #         spec.bin.update(hns=vals.shape[1])
        
    # qx_project = np.linspace(qx[0], qx[-1], len(qx))
    # qy_project = np.linspace(qy[0], qy[-1], len(qy))
    if smooth_segy:
        copyfile(seismic_data_path, smooth_segy_file)
        with segyio.open(smooth_segy_file, 'r+', strict=False) as dst:
            c = 0
            for ix, header in enumerate(dst.header):
                if ix < len(qx):
                    header.update({segyio.TraceField.GroupX: int(qx[ix] * 10000)})
                    header.update({segyio.TraceField.GroupY: int(qy[ix] * 10000)})
                    header.update({segyio.TraceField.SourceX: int(qx[ix] * 10000)})
                    header.update({segyio.TraceField.SourceY: int(qy[ix] * 10000)})
                    header.update({segyio.TraceField.CDP_X: int(qx[ix] * 10000)})
                    header.update({segyio.TraceField.CDP_Y: int(qy[ix] * 10000)})
                else:
                    header.update({segyio.TraceField.GroupX: int(qx[-1] * 10000 + c)})
                    header.update({segyio.TraceField.GroupY: int(qy[-1] * 10000 + c)})
                    header.update({segyio.TraceField.SourceX: int(qx[-1] * 10000 + c)})
                    header.update({segyio.TraceField.SourceY: int(qy[-1] * 10000 + c)})
                    header.update({segyio.TraceField.CDP_X: int(qx[-1] * 10000 + c)})
                    header.update({segyio.TraceField.CDP_Y: int(qy[-1] * 10000 + c)})
                    c += 10
        # with segyio.open(seismic_data_path, strict=False) as src:
        #     spec=segyio.tools.metadata(src)
        #     with segyio.create(smooth_segy_file, spec) as dst:
        #         dst.text[0] = src.text[0]
        #         dst.bin = src.bin
        #         spec.samples = spec.samples * depth_conversion_velocity / 2
        #         cutoff = np.argmin(np.abs(spec.samples - zlim[1] * 1000))
        #         spec.samples = spec.samples[:cutoff]
        #         dst.bin.update(hns=len(spec.samples))
        #         dst.header = src.header
        #         dst.trace = src.trace
    if project_segy:
        copyfile(seismic_data_path, projected_segy_file)
        with segyio.open(projected_segy_file, 'r+', strict=False) as dst:
            c = 0
            for ix, header in enumerate(dst.header):
                if ix < len(qx):
                    header.update({segyio.TraceField.GroupX: int(qx_project[ix] * 10000)})
                    header.update({segyio.TraceField.GroupY: int(qy_project[ix] * 10000)})
                    header.update({segyio.TraceField.SourceX: int(qx_project[ix] * 10000)})
                    header.update({segyio.TraceField.SourceY: int(qy_project[ix] * 10000)})
                    header.update({segyio.TraceField.CDP_X: int(qx_project[ix] * 10000)})
                    header.update({segyio.TraceField.CDP_Y: int(qy_project[ix] * 10000)})
                else:
                    header.update({segyio.TraceField.GroupX: int(qx_project[-1] * 10000 + c)})
                    header.update({segyio.TraceField.GroupY: int(qy_project[-1] * 10000 + c)})
                    header.update({segyio.TraceField.SourceX: int(qx_project[-1] * 10000 + c)})
                    header.update({segyio.TraceField.SourceY: int(qy_project[-1] * 10000 + c)})
                    header.update({segyio.TraceField.CDP_X: int(qx_project[-1] * 10000 + c)})
                    header.update({segyio.TraceField.CDP_Y: int(qy_project[-1] * 10000 + c)})
                    c += 10
    if project_model:
        qx_project = np.linspace(qx[0], qx[-1], len(qx))
        qy_project = np.linspace(qy[0], qy[-1], len(qy))
        if offset_model:
            model_offset = 500
        # segyio.tools.from_array(smooth_model_segy, vals, dt=dz)
        with segyio.open(seismic_data_path, 'r', strict=False) as src:
            xdiff = abs(qx_project[0] - qx_project[-1])
            ydiff = abs(qy_project[0] - qy_project[-1])
            if xdiff > ydiff:
                x_offset = 500
                y_offset = 0
            else:
                y_offset = 500
                x_offset = 0
            # dt_new = int(round(2 * 1000 * 1000 * dz / depth_conversion_velocity))
            dt_new = int(dz * 1000)
            spec=segyio.tools.metadata(src)
            # spec.samples = np.arange(len(qz)) * dt_new / 1000
            spec.samples = np.arange(len(qz)) * dt_new
            with segyio.create(projected_model_segy, spec) as dst:
                dst.text[0] = src.text[0]
                dst.bin = src.bin
                dst.bin.update(hns=len(spec.samples))
                dst.bin.update(hdt=dt_new)
                dst.header = src.header
                dst.trace = src.trace

                c = 0
                vals_test = np.zeros(vals.shape)
                for ix, trace in enumerate(dst.trace):
                    if ix < vals.shape[0]:
                        dst.trace[ix] = vals[ix, :]
                        vals_test[ix, :] = dst.trace[ix]
                    else:
                        dst.trace[ix] = vals[-1, :]
                        vals_test[ix, :] = dst.trace[ix]
                for ix, header in enumerate(dst.header):
                    header.update({segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_new})
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
    if smooth_model:
        # segyio.tools.from_array(smooth_model_segy, vals, dt=dz)
        with segyio.open(seismic_data_path, 'r', strict=False) as src:
            dt_new = int(round(2 * 1000 * 1000 * dz / depth_conversion_velocity))
            spec=segyio.tools.metadata(src)
            spec.samples = np.arange(len(qz)) * dt_new / 1000
            with segyio.create(smooth_model_segy, spec) as dst:
                dst.text[0] = src.text[0]
                dst.bin = src.bin
                dst.bin.update(hns=len(spec.samples))
                dst.bin.update(hdt=dt_new)
                dst.header = src.header
                dst.trace = src.trace

                c = 0
                for ix, trace in enumerate(dst.trace):
                    if ix < vals.shape[0]:
                        dst.trace[ix] = vals[ix, :]
                    else:
                        dst.trace[ix] = vals[-1, :]
                for ix, header in enumerate(dst.header):
                    header.update({segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_new})
                    if ix < len(qx):
                        header.update({segyio.TraceField.GroupX: int(qx[ix] * 10000)})
                        header.update({segyio.TraceField.GroupY: int(qy[ix] * 10000)})
                        header.update({segyio.TraceField.SourceX: int(qx[ix] * 10000)})
                        header.update({segyio.TraceField.SourceY: int(qy[ix] * 10000)})
                        header.update({segyio.TraceField.CDP_X: int(qx[ix] * 10000)})
                        header.update({segyio.TraceField.CDP_Y: int(qy[ix] * 10000)})
                    else:
                        header.update({segyio.TraceField.GroupX: int(qx[-1] * 10000 + c)})
                        header.update({segyio.TraceField.GroupY: int(qy[-1] * 10000 + c)})
                        header.update({segyio.TraceField.SourceX: int(qx[-1] * 10000 + c)})
                        header.update({segyio.TraceField.SourceY: int(qy[-1] * 10000 + c)})
                        header.update({segyio.TraceField.CDP_X: int(qx[-1] * 10000 + c)})
                        header.update({segyio.TraceField.CDP_Y: int(qy[-1] * 10000 + c)})
                        c += 10