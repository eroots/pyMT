#!/usr/bin/env python
import os


def search_dir(suffix):
    lst = []
    for file in os.listdir():
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
    else:
        suffix = 'all'
        ending = 'AML'
    lst = [file for file in lst if file[-5] in ending]
    with open(''.join([transect, suffix, '.lst']), 'w') as f:
        f.write(str(len(lst)) + '\n')
        for file in lst:
            f.write(file + '\n')


if __name__ == '__main__':
    suffix = ['DRY', 'ATT', 'RRV', 'STU', '']
    for suff in suffix:
        lst = search_dir(suff)
        for data_type in ['amt', 'bb', 'all']:
            write_list(suff, lst, data_type)
