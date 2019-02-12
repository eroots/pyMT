import pyMT.data_structures as WSDS
import pyMT.utils as utils
import numpy as np


def station_misfit(data, resp):
    misfit = {comp: [] for comp in data.components}
    misfit.update({'Total': 0})
    for comp in misfit.keys():
        if comp in data.components:
            misfit[comp] = (np.abs(resp.data[comp] - data.data[comp]) / data.errors[comp]) ** 2
            misfit['Total'] += misfit[comp] ** 2
            misfit[comp] = np.sqrt(misfit[comp])
    misfit['Total'] = np.sqrt(misfit['Total'] / len(misfit.keys()))
    return misfit


def dataset_misfit(dataset):
    site_names = ds.data.site_names
    misfit = {site: [] for site in site_names}
    for site in site_names:
        misfit[site] = station_misfit(ds.data.sites[site], ds.response.sites[site])
    return misfit


listfile = 'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/MetalEarth/sturgeon/j2/allsites.lst'
datafile = 'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/MetalEarth/sturgeon/stu3/stu2_j2Rot2.data'
respfile = 'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/MetalEarth/sturgeon/stu3/stu3_final.resp'
ds = WSDS.Dataset(listfile=listfile, datafile=datafile, responsefile=respfile)
misfit = dataset_misfit(ds)
