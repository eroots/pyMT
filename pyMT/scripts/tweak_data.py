import pyMT.data_structures as DS
import numpy as np
from copy import deepcopy


list_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst'
data_file1 = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/wst_cullmantle_all_base.dat'
data_file2 = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/wst_cullmantle_all_regged.dat'

dataset_base = DS.Dataset(listfile=list_file, datafile=data_file1)

dataset = DS.Dataset(listfile=list_file, datafile=data_file2)


base_errors = {site: {comp: deepcopy(dataset_base.data.sites[site].used_error[comp]) for comp in dataset_base.data.components}
                      for site in dataset_base.data.site_names}
dataset.data.error_floors['Tipper'] = 0.03
dataset.data.apply_error_floor()
dataset.regulate_errors(multiplier=1.2, fwidth=0.7)
dataset.data.write('E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/wst_cullmantle_all_regged.dat')
for site in dataset.data.site_names:
    for comp in dataset.data.components:
        base_errors[site].update({comp: dataset.data.sites[site].used_error[comp]})
        # Flag the FFMT removed data
        idx = dataset.data.sites[site].data[comp] == 0
        if np.any(idx):
            dataset.data.sites[site].used_error[comp][idx] = dataset.data.REMOVE_FLAG
            dataset.data.sites[site].errors[comp][idx] = dataset.data.REMOVE_FLAG
        if comp.lower().startswith('t'):
            # Flag broadband tipper data
            # if np.all(dataset.data.sites[site].used_error[comp][-3:] == dataset.data.REMOVE_FLAG):
            if dataset.raw_data.sites[site].periods[0] < 1:
                dataset.data.sites[site].used_error[comp][:] = dataset.data.REMOVE_FLAG
                dataset.data.sites[site].errors[comp][:] = dataset.data.REMOVE_FLAG
            else:
                # Flag tipper data at periods > 7500
                pdx = dataset.data.periods > 7500
                # idx = dataset.data.sites[site].data == 0
                dataset.data.sites[site].used_error[comp][pdx] = dataset.data.REMOVE_FLAG
                dataset.data.sites[site].errors[comp][pdx] = dataset.data.REMOVE_FLAG
                # dataset.data.sites[site].used_error[comp][idx] = dataset.data.REMOVE_FLAG
                # dataset.data.sites[site].errors[comp][idx] = dataset.data.REMOVE_FLAG
        dataset.data.sites[site].used_error[comp] = np.max([dataset.data.sites[site].used_error[comp],
                                                           base_errors[site][comp]], axis=0)

dataset.data.write('E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/wst_cullmantle_all')

# Set up and write out the files for XY and YX inversions.
names = ('XY', 'YX', 'K')
for ii, setup in enumerate([['ZXXR', 'ZXXI', 'ZXYR', 'ZXYI'], ['ZYYR', 'ZYYI', 'ZYXR', 'ZYXI'], ['TZXR', 'TZXI', 'TZYR', 'TZYI']]):
    data2 = DS.Data('E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/wst_cullmantle_all.dat')
    for site in data2.site_names:
        for comp in data2.components:
            if comp not in setup:
                data2.sites[site].used_error[comp][:] = data2.REMOVE_FLAG
    if ii == 2:
        data2.inv_type = 3
    else:
        data2.inv_type = 1

    data2.write('E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/wst_cullmantle3_{}.dat'.format(names[ii]))


# Regulate errors (conservatively)

# Go back and set errors to the higher of base (raw) and regulated errors

# for site in dataset.data.site_names:
#     for comp in dataset


