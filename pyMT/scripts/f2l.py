#!/usr/bin/env python
import os


def search_dir(suffix):
    lst = []
    for file in os.listdir():
        if suffix in file and file.endswith('edi'):
            lst.append(file)
    return lst


def write_list(suffix, lst):
    with open(''.join(['allsites', suffix, '.lst']), 'w') as f:
        f.write(str(len(lst)) + '\n')
        for file in lst:
            f.write(file + '\n')


if __name__ == '__main__':
    suffix = ['DRY', 'ATT', 'RRV', 'STU', '']
    for suff in suffix:
        lst = search_dir(suff)
        write_list(suff, lst)
