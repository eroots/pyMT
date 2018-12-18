import pyMT.data_structures as WSDS
import pandas as pd
import numpy as np


# Define your data and list locations
data_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/finish/swz_cull1i_Z.dat'
list_file = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/swz_cull1.lst'
# Read data and list
data = WSDS.Data(data_file)
raw = WSDS.RawData(list_file)
# Define what info you want in the CSV (headers)
columns = ('site', 'X', 'Y', 'period', 'beta', 'alpha',
           'det_phi', 'skew_phi', 'phi_1', 'phi_2', 'phi_3',
           'phi_max', 'phi_min', 'Lambda', 'azimuth', 'delta')


c = 0
site_list = []
# Initialize numpy array with zeros
arr = np.zeros((len(data.site_names) * len(data.periods), len(columns)))
for site in data.site_names:
    for ii, period in enumerate(data.periods):
        phase_tensor = data.sites[site].phase_tensors[ii]
        for jj, column in enumerate(columns):
            # Can't pass string to np array, so ignore for now
            if column == 'site':
                # arr[ii, jj] = site
                pass
            # If looking for lat/longs, grab these from the RawData object
            elif column == 'X':
                arr[c, jj] = raw.sites[site].locations['Long']
            elif column == 'Y':
                arr[c, jj] = raw.sites[site].locations['Lat']
            # Otherwise just pull the info from the phase_tensor object
            else:
                arr[c, jj] = getattr(phase_tensor, column)
        c += 1
        site_list.append(site)

df = pd.DataFrame(data=arr, columns=columns)
df['site'] = site_list
df.to_csv('outfile.csv')
