from scipy.sparse import csr_matrix as csr
import numpy as np

csr_file = 'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/MARE2DEM/synth_test/small_test.penalty'
res_file = 'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/MARE2DEM/synth_test/small_test.0.resistivity'

with open(file, 'r') as f:
    next(f)
    nnz, P = [int(x) for x in next(f).strip().split()]
    lines = f.readlines()
    mat = np.loadtxt(lines[:nnz])
    cols = mat[:, 0] - 1
    vals = mat[:, 1]
    row_idx = np.loadtxt(lines[nnz:]) - 1
    sparse = csr((vals, cols, row_idx), shape=(P, 16))

