import numpy as np
import sys
import pyMT.utils as utils


def read_occam_data(datafile):
    with open(datafile, 'r') as f:
        lines = f.readlines()
        for ii, line in enumerate(lines):
            if 'DATA BLOCKS:' in line:
                break
        ii += 2
        headings = lines[:ii]
        data = np.loadtxt(lines[ii:])
        return data, headings


def add_noise(d, perc_error):
    if perc_error > 1:
        perc_error /= 100
    log_noise = perc_error / np.log(10)
    dnew = d
    idx = np.bitwise_or(d[:, 2] == 1, d[:, 2] == 5)
    dnew[idx, 3] = d[idx, 3] + np.random.normal(scale=log_noise, size=np.sum(idx))
    idx = np.bitwise_or(d[:, 2] == 2, d[:, 2] == 6)
    dnew[idx, 3] += np.random.normal(scale=100 * log_noise / 2, size=np.sum(idx))
    return dnew


def add_shift(d):
    nsites = np.unique(d[:, 0])
    TE_shifts = np.random.uniform(low=-1, high=1, size=len(nsites))
    TM_shifts = np.random.uniform(low=-1, high=1, size=len(nsites))
    for site in range(len(nsites)):
        idx = np.bitwise_and(d[:, 0] == site, d[:, 2] == 1)
        d[idx, 3] += TE_shifts[site]
        idx = np.bitwise_and(d[:, 0] == site, d[:, 2] == 5)
        d[idx, 3] += TM_shifts[site]
    return d


def write_occam_data(d, headings, datafile):
    with open(datafile, 'w') as f:
        f.writelines(headings)
        for line in d:
            f.write('{:>5d} {:>5d} {:>5d} {:>18.5f} {:>18.4E}\n'.format(
                    int(line[0]), int(line[1]), int(line[2]), line[3], line[4]))


def main():
    if len(sys.argv) < 3:
        print('Command line arguments are: <data file (string)> ' +
              ' <percentage noise (double)> <static shift (binary)')
        return
    if not utils.check_file(sys.argv[1]):
        print('File {} not found'.format(sys.argv[1]))
        return
    else:
        datafile = sys.argv[1]
    if len(sys.argv) > 1:
        perc_error = float(sys.argv[2])
    else:
        perc_error = 5.0
    if len(sys.argv) > 2:
        static_shift = int(sys.argv[3])
    else:
        static_shift = 0
    data, headings = read_occam_data(datafile)
    if perc_error > 0:
        new_data = add_noise(data, perc_error)
    else:
        print('Not adding noise...')
        new_data = data
    if static_shift:
        new_data = add_shift(new_data)
    write_occam_data(new_data, headings, datafile + '_noise')


if __name__ == '__main__':
    main()
