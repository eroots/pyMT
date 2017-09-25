import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_data(data, response, sites=None):
    if not sites:
        sites = list(response.keys())
    jj = 0
    figc = 0
    figs = []
    figs.append(plt.figure(figsize=(26, 15)))
    for ii, site in enumerate(sites):
        jj += 1
        if jj == 4:
            jj = 1
            figs.append(plt.figure(figsize=(26, 15)))
            figc += 1
        ax = figs[figc].add_subplot(2, 3, jj)
        ax.set_title(site + ': Rho')
        ax.errorbar(np.log10(data[site]['Freqs']), data[site]['RhoXY'], color='r', linestyle='',
                    marker='o', label='XY', xerr=None, yerr=data[site]['RhoXY_errs'])
        ax.errorbar(np.log10(data[site]['Freqs']), data[site]['RhoYX'], color='g', linestyle='',
                    marker='o', label='YX', xerr=None, yerr=data[site]['RhoYX_errs'])
        ax.plot(np.log10(response[site]['Freqs']), response[site]['RhoXY'], 'r-')
        ax.plot(np.log10(response[site]['Freqs']), response[site]['RhoYX'], 'g-')
        if jj == 1:
            ax.legend()
        ax = figs[figc].add_subplot(2, 3, jj + 3)
        ax.set_title(site + ': Phase')
        ax.errorbar(np.log10(data[site]['Freqs']), data[site]['PhaXY'], color='r', linestyle='',
                    marker='o', label='XY', xerr=None, yerr=data[site]['PhaXY_errs'])
        ax.errorbar(np.log10(data[site]['Freqs']), data[site]['PhaYX'], color='g', linestyle='',
                    marker='o', label='YX', xerr=None, yerr=data[site]['PhaYX_errs'])
        ax.plot(np.log10(response[site]['Freqs']), response[site]['PhaXY'], 'r-')
        ax.plot(np.log10(response[site]['Freqs']), response[site]['PhaYX'], 'g-')
        if jj == 1:
            ax.legend()
    # plt.show()
    return figs


def read_data(datafile):
    with open(datafile, 'r') as f:
        lines = f.readlines()
        for ii, line in enumerate(lines):
            if 'sites' in line.lower():
                NS = int(line.split(':')[1].strip())
                site_line = ii + 1
            if 'frequencies' in line.lower():
                NF = int(line.split(':')[1].strip())
                freq_line = ii + 1
                break
        frequencies = []
        for line in lines[freq_line:freq_line + NF]:
            if 'DATA' in line.upper():
                break
            else:
                frequencies.append(line)
        frequencies = [x.split() for x in frequencies]
        frequencies = [item for sublist in frequencies for item in sublist]
        frequencies = [float(x) for x in frequencies]
        frequencies = np.array([float(f) for f in frequencies])
        sites = lines[site_line: site_line + NS]
        sites = [site.strip() for site in sites]
    return sites, frequencies


def read_response(respfile, sites, freqs):
    data = np.loadtxt(respfile)
    site_nums = data[:, 0]
    freq_num = data[:, 1]
    d_type = data[:, 2]
    data_points = data[:, 4]
    resp_points = data[:, 5]
    n_errors = data[:, 6]
    error_bars = (data_points - resp_points) / n_errors
    all_data = {}
    all_resp = {}
    for ii, site in enumerate(sites):
        idx = site_nums == ii + 1
        site_freqs = (np.unique(freq_num[idx])) - 1
        actual_freqs = 1 / freqs[site_freqs.astype(int)]
        rhoXY_idx = d_type == 1
        rhoYX_idx = d_type == 5
        phaXY_idx = d_type == 2
        phaYX_idx = d_type == 6
        rhoXY_data = data_points[np.bitwise_and(idx, rhoXY_idx)]
        rhoXY_resp = resp_points[np.bitwise_and(idx, rhoXY_idx)]
        rhoXY_errs = error_bars[np.bitwise_and(idx, rhoXY_idx)]
        rhoYX_data = data_points[np.bitwise_and(idx, rhoYX_idx)]
        rhoYX_resp = resp_points[np.bitwise_and(idx, rhoYX_idx)]
        rhoYX_errs = error_bars[np.bitwise_and(idx, rhoYX_idx)]
        phaXY_data = data_points[np.bitwise_and(idx, phaXY_idx)]
        phaXY_resp = resp_points[np.bitwise_and(idx, phaXY_idx)]
        phaXY_errs = error_bars[np.bitwise_and(idx, phaXY_idx)]
        phaYX_data = data_points[np.bitwise_and(idx, phaYX_idx)]
        phaYX_resp = resp_points[np.bitwise_and(idx, phaYX_idx)]
        phaYX_errs = error_bars[np.bitwise_and(idx, phaYX_idx)]

        RMS = {'rhoXY': np.sqrt(np.sum(n_errors[np.bitwise_and(idx, rhoXY_idx)] ** 2)),
               'rhoYX': np.sqrt(np.sum(n_errors[np.bitwise_and(idx, rhoYX_idx)] ** 2)),
               'phaXY': np.sqrt(np.sum(n_errors[np.bitwise_and(idx, phaXY_idx)] ** 2)),
               'phaYX': np.sqrt(np.sum(n_errors[np.bitwise_and(idx, phaYX_idx)] ** 2))}

        all_data.update({site: {'Number': ii + 1, 'Freqs': actual_freqs,
                                'RhoXY': rhoXY_data, 'RhoYX': rhoYX_data,
                                'PhaXY': phaXY_data, 'PhaYX': phaYX_data,
                                'RhoXY_errs': rhoXY_errs, 'RhoYX_errs': rhoYX_errs,
                                'PhaXY_errs': phaXY_errs, 'PhaYX_errs': phaYX_errs}})
        all_resp.update({site: {'Number': ii + 1, 'Freqs': actual_freqs,
                                'RhoXY': rhoXY_resp, 'RhoYX': rhoYX_resp,
                                'PhaXY': phaXY_resp, 'PhaYX': phaYX_resp,
                                'RMS': RMS}})
    return all_data, all_resp


def main():
    datafile = None
    respfile = None
    for file_in in sys.argv:
        if 'resp' in file_in.lower():
            respfile = file_in
        elif 'data' in file_in.lower():
            datafile = file_in
    if not respfile or not datafile:
        print('Must specify both data and response files...')
        return
    else:
        sites, freqs = read_data(datafile)
        data, resp = read_response(respfile, sites, freqs)
        figs = plot_data(data, resp, sites)
        return figs


if __name__ == '__main__':
    figs = main()
