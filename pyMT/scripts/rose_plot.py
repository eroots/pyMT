import numpy as np
import matplotlib.pyplot as plt
import pyMT.data_structures as WSDS


def gen_tiling(num_plots):
        if num_plots < 3:
            tiling = [num_plots, 1]
        else:
            s1 = np.floor(np.sqrt(num_plots))
            s2 = np.ceil(num_plots / s1)
            tiling = [int(s1), int(s2)]
        return tiling


# data = WSDS.Data(datafile='C:/Users/eroots/phd/ownCloud/data/Regions/afton/afton1/afton_cull1.dat',
#                  listfile='C:/Users/eroots/phd/ownCloud/data/Regions/afton/j2/afton_cull1.lst')
# data = WSDS.Data(datafile='C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/malartic/mal1/moresites/finish/mal5_all.dat',
#                  listfile='C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal5.lst')
# data = WSDS.RawData(listfile='C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/matheson/j2/mat_westLine.lst')
data = WSDS.Data('C:/Users/eroots/phd/ownCloud/data/Regions/afton/sorted_lines.dat')
# data = WSDS.Data('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/R2southeast_1/2d/R2SE_2d_0deg.dat')
# NP = len(data.master_period_list().keys())
NP = data.NP
plots_per_fig = 6
N = 45
bottom = 0
max_height = 4
plot_range = (-np.pi / 2, np.pi / 2)
width = np.pi / N
theta = np.linspace(plot_range[0], plot_range[1], N, endpoint=False)

figures = {}
figures_TF = {}
p_idx = 0
tiling = gen_tiling(plots_per_fig)
# for ii in range(int(np.ceil(data.NP / plots_per_fig))):
for ii in range(int(np.ceil(NP / plots_per_fig))):
    figures.update({ii: plt.figure()})
    axes = []
    for jj in range(plots_per_fig):
        # if p_idx < data.NP:
        if p_idx < NP:
            azimuths = np.array([data.sites[site].phase_tensors[p_idx].azimuth for site in data.site_names])
            radii = np.histogram(azimuths, range=plot_range, bins=N)[0]
            axes.append(figures[ii].add_subplot(tiling[0], tiling[1], jj + 1, polar=True))
            axes[jj].bar(theta, radii, width=width, bottom=bottom)
            # axes[jj].set_title('Period: {:>5.4g} s'.format(data.periods[p_idx]))
            axes[jj].set_title('Period: {:>5.4g} s'.format(data.sites[data.site_names[0]].periods[p_idx]))
            p_idx += 1
# p_idx = 0
# for ii in range(int(np.ceil(data.NP / plots_per_fig))):
#     figures_TF.update({ii: plt.figure()})
#     axes = []
#     for jj in range(min(plots_per_fig, data.NP - p_idx)):
#         TY = np.array([data.sites[site].data['TZYR'][p_idx] for site in data.site_names])
#         TX = np.array([data.sites[site].data['TZXR'][p_idx] for site in data.site_names])
#         azimuths = (np.arctan(TY / TX))
#         radii = np.histogram(azimuths, range=plot_range, bins=N)[0]
#         axes.append(figures_TF[ii].add_subplot(tiling[0], tiling[1], jj + 1, polar=True))
#         axes[jj].bar(theta, radii, width=width, bottom=bottom)
#         axes[jj].set_title('Period: {:>5.4g} s'.format(data.periods[p_idx]))
#         p_idx += 1
plt.show()
