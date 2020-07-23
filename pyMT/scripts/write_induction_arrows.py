import pyMT.data_structures as DS
import numpy as np


# Use only one of 'list_file' or 'data_file'. Comment out the appropriate line, as well as the corresponding
# 'RawData' or 'Data' line.
# Since the range of possible tipper magnitudes is fairly wide (an order of magnitude)
# compared to phase tensor major / minor axes (roughly a factor of 2 or 3 at most),
# low magnitude arrows can be much smaller than higher magnitude arrows. 
# Feel free to play around with normalizing arrow lengths.
                                #

list_file = '/your/list/file'    # List file containing sites to write out
data_file = '/your/data/file'   # Data file to use
out_file = '/your/output/file'  # Outfile file name
verbose = True                  # False only write info needed for real arrows, True write imaginary component as well.
scale_factor = 1/2500           # Size scale factor for arrows, measured as a fraction of diagonal window size
                                # e.g., if your stations cover 60 km in X and 60 km in Y (window size of ~85 km),
                                # a scale_factor = 1/50 gives a magnitude 1 induction arrow a length of ~1.7 km
                                # Note that scales are tricky for arrows. Play around with different values.


# data = DS.RawData(list_file)
data = DS.Data(data_file)


if not out_file.endswith('.csv'):
    out_file += '.csv'
print('Writing induction arrow data to {}'.format(out_file))
with open(out_file, 'w') as f:
    header = ['Site', 'Period', 'Latitude', 'Longitude', 'Azimuth_Real',
              'Magnitude_Real', 'Magnitude_Real_scaled']
    if verbose:
        header += ['Azimuth_Imag', 'Imag_Magnitude', 'Imag_Magnitude_scaled']
    f.write(','.join(header))
    f.write('\n')
    try:
        test = data.sites[data.site_names[0]].locations['Lat'], data.sites[data.site_names[0]].locations['Long']
        X_key, Y_key = 'Lat', 'Long'
    except KeyError:
        X_key, Y_key = 'Y', 'X'
    X_all = [site.locations['X'] for site in data.sites.values()]
    Y_all = [site.locations['Y'] for site in data.sites.values()]
    scale = np.sqrt((np.max(X_all) - np.min(X_all)) ** 2 +
                    (np.max(Y_all) - np.min(Y_all)) ** 2)
    scale *= scale_factor
    for site_name in data.site_names:
        site = data.sites[site_name]
        X, Y = site.locations[X_key], site.locations[Y_key]
        for ii, period in enumerate(site.periods):
            azimuth_real = np.angle(-1 * site.data['TZYR'][ii] + 1j * -1 * site.data['TZXR'][ii], deg=True)
            mag_real = np.sqrt(site.data['TZYR'][ii] ** 2 + site.data['TZXR'][ii] ** 2)
            mag_real_scaled = mag_real * scale  # Change things here if you want to play with scaling
            f.write('{}, {}, {}, {}, {}, {}, {}'.format(site_name,
                                                    period,
                                                    Y,
                                                    X,
                                                    azimuth_real,
                                                    mag_real,
                                                    mag_real_scaled))
            if verbose:
                azimuth_imag = np.angle(-1 * site.data['TZYI'][ii] + 1j * -1 * site.data['TZXI'][ii], deg=True)
                mag_imag = np.sqrt(site.data['TZYI'][ii] ** 2 + site.data['TZXI'][ii] ** 2)
                mag_imag_scaled = mag_imag * scale # Change things here for imaginary arrows.
                f.write(', {}, {}, {}\n'.format(azimuth_imag,
                                                mag_imag,
                                                mag_imag_scaled))
            else:
                f.write('\n')