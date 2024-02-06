from netcdf_segy.segy2netcdf import segy2netcdf
import segyio
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import os

# E:\phd\NextCloud\Documents\ME_Transects\Upper_Abitibi\Paper\RoughFigures
# seisline = ['12','14','15','16','16a','23','25','28','17','18','21','24','27']
seisline = ['16']
# seisline += ['17','18','21','24','27']
# line = 12
save_fig = 1
linear_xaxis = 0
use_cdps = 1
save_path = 'E:/phd/NextCloud/Documents/ME_Transects/Upper_Abitibi/Paper/RoughFigures/t2d-seismic/'
dpi = 600
aspect_ratio = 1
downsample = 0

plot_direction = 'lr'


# copyfile(segy_file, 'temp.sgy')
for line in seisline:
    segy_file = 'D:/OpenDTect_data/RawData/output/line{}_depth.sgy'.format(line)
    save_file = 'line{}_depth_cdp.png'.format(line)
    with segyio.open(segy_file, 'r', strict=False) as src:
        rate = segyio.tools.dt(src)
        # segyio.tools.resample(src, rate=int(round(rate/4)))
        x = np.array([t[segyio.TraceField.CDP_X] for t in src.header])
        y = np.array([t[segyio.TraceField.CDP_Y] for t in src.header])
        cdp = np.array([t[segyio.TraceField.CDP] for t in src.header])
        nt = len(x)
        # seis_data = np.zeros((src.header[0][segyio.TraceField.TRACE_SAMPLE_COUNT], nt))
        seis_data = src.trace.raw[:]
        xdiff = np.abs(x[0] - x[-1])
        ydiff = np.abs(y[0] - y[-1])
        if xdiff > ydiff:
            header1, val1 = 'CDP_Y', 1
            header2, val2 = 'CDP_X', nt
        else:
            header1, val1 = 'CDP_X', 1
            header2, val2 = 'CDP_Y', nt
        samples = src.samples
        if downsample:
            spec = segyio.tools.metadata(src)
            spec.samples = spec.samples[::4]
            with segyio.create('temp.sgy', spec) as dst:
                # segyio.tools.resample(dst, rate=int(round(rate/4)))
                dst.text[0] = src.text[0]
                dst.bin = src.bin
                dst.header = src.header
                # dst.trace = src.trace
                dst.bin.update(hns=int(len(spec.samples) / 4))
                dst.bin.update(hdt=int(rate*4))

                # for ix, trace in enumerate(src.trace):
                #     # seis_data[:, ix] = src.trace[ix]
                #     dst.trace[ix] = seis_data[ix, ::4]
                for ix, header in enumerate(dst.header):
                    header.update({segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(rate * 4)})
                    header.update({segyio.TraceField.TRACE_SAMPLE_COUNT: int(seis_data.shape[1] / 4)})
            segy2netcdf('temp.sgy',
                        'D:/OpenDTect_data/RawData/netcdf/line{}_depth_resample.nc'.format(line), 'Depth', compress=False,
                         d=((header1, val1), (header2, val2)))

        else:
            copyfile(segy_file, 'temp.sgy')

    if save_fig:
        seis_data = seis_data.T
        if linear_xaxis:
            linear_x = np.zeros(x.shape)
            linear_x[1:] = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
            linear_x = np.cumsum(linear_x)
            nodes = np.array([y * 1000, x * 1000]).T
            # linear_site = np.zeros((len(data.locations)))
            # for ii, (x, y) in enumerate(data.locations):
                # if use_nudge and data.site_names[ii] in nudge_sites:
                    # y += nudge_dist
                # dist = np.sum((nodes - np.array([x, y])) ** 2, axis=1)
                # idx = np.argmin(dist)
                # linear_site[ii] = linear_x[idx]
            x_axis = linear_x
            x_axis_label = 'Distance (km)'
            if xdiff > ydiff:
                if x[0] > x[-1]:
                    seis_data = np.fliplr(seis_data)
            else:
                if y[0] > y[-1]:
                    seis_data = np.fliplr(seis_data)
        elif use_cdps:
            x_axis = range(len(cdp))
            x_axis_label = 'CDP #'
            aspect_ratio = 1/50
            if xdiff > ydiff:
                if x[0] > x[-1]:
                    seis_data = np.fliplr(seis_data)
                    x_axis = np.flip(x_axis, axis=0)
            else:
                if y[0] > y[-1]:
                    seis_data = np.fliplr(seis_data)
                    x_axis = np.flip(x_axis, axis=0)
        else:
            if xdiff > ydiff:
                x_axis_label = 'Easting (km)'
                x_axis = x
                if x[0] > x[-1]:
                    seis_data = np.fliplr(seis_data)
                    x_axis = np.flip(x_axis, axis=0)
            else:
                x_axis_label = 'Northing (km)'
                x_axis = y
                if y[0] > y[-1]:
                    seis_data = np.fliplr(seis_data)
                    x_axis = np.flip(x_axis, axis=0)
        # seis_data = seis_data / np.linalg.norm(seis_data, axis=0)
        seis_data = seis_data / np.linalg.norm(seis_data, axis=0)
        # seis_data = np.fliplr(seis_data)
        seis_data[np.isnan(seis_data)] = 0
        clip_val = 0.5*np.max(np.abs(seis_data))
        seis_data[seis_data<-clip_val] = -clip_val
        seis_data[seis_data>clip_val] = clip_val
        # tvec = np.arange(seis_data.shape[0]) * dt / 1000
        # dvec = tvec * depth_conversion_velocity / 2
        x_axis = x_axis / 100000
        plt.imshow(np.abs(seis_data), 
                   cmap='gray_r',
                   extent=[x_axis[0], x_axis[-1], samples[-1]/1000, samples[0]/1000],
                   alpha=1)
        plt.gcf().set_size_inches(12, 8)
        plt.gca().set_aspect(aspect_ratio)
        plt.gca().set_xlabel(x_axis_label)
        plt.gca().set_ylabel('Depth (km)')
        plt.savefig(save_path + save_file, dpi=dpi,
                                transparent=True, bbox_inches='tight')
        # fig.clear()
        # ax.clear()
        # fig.gcf()
        plt.clf()
        plt.cla()
        plt.close('all')
        # plt.pause(1)

    os.remove('temp.sgy')