#!/c/Users/eric/Anaconda3/python
import sys
import os


inputs = sys.argv[1:]
print(inputs)
if len(inputs) == 0:
    ext = '.dat'
    contains = ''
elif len(inputs) == 1:
    ext = inputs[0]
    contains = ''
elif len(inputs) == 2:
    ext = inputs[0]
    contains = inputs[1]

files = os.listdir()
lst = []
for file in files:
    if file.endswith(ext) and contains in file:
        lst.append(file)

with open(''.join(['allsites', contains, '.lst']), 'w') as f:
    f.write(str(len(lst)) + '\n')
    for file in lst:
        f.write(file + '\n')


