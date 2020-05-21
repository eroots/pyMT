import pyMT.data_structures as DS

listfile = 'your/listfile/here' # List file containing sites to write out
outfile = 'your/outfile/here'   # Outfile file name
verbose = True					# False only write info needed for ellipses, True write additional PT parameters
scale_factor = 1/50				# Size scale factor for ellipses, measured as a fraction of diagonal window size
								# e.g., if your stations cover 60 km in X and 60 km in Y (window size of ~85 km),
								# a scale_factor = 1/50 gives AN ellipse radius of ~1.7 km

data = DS.RawData(listfile)
data.write_phase_tensors(out_file=outfile, verbose=verbose, scale_factor=scale_factor)