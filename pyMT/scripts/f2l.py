#!/usr/bin/env python
import os
import sys

def search_dir(suffix):
    lst = []
    for file in os.listdir():
        print(file)
        if suffix in file and file.endswith('edi'):
            lst.append(file)
    return lst


def write_list(transect, lst, data_type):
    if data_type.lower() == 'amt':
        suffix = 'AMT'
        ending = 'A'
    elif data_type.lower() == 'bb':
        suffix = 'BB'
        ending = 'ML'
    elif data_type.lower() == 'all':
        suffix = 'all'
        ending = 'AML'
    else:
        suffix = ''
        ending = ''
    if data_type:
        lst = [file for file in lst if file[-5] in ending]
    else:
        lst = [file for file in lst if file.endswith('edi')]
    with open(''.join([transect, suffix, '.lst']), 'w') as f:

        f.write(str(len(lst)) + '\n')
        for file in lst:
            f.write(file + '\n')


if __name__ == '__main__':
    if sys.argv[1].lower() == 'simple':
        suffix = ''
        data_types = ''
    else:
        suffix = ['DRY', 'ATT', 'RRV', 'STU',
                  'CHI', 'COB', 'GER', 'LAR',
                  'MAL', 'MAT', 'ROU', 'SUD', 'SWZ', '']
        data_types = ['amt', 'bb', 'all']
    for suff in suffix:
        lst = search_dir(suff)
        for data_type in data_types:
            write_list(suff, lst, data_type)
