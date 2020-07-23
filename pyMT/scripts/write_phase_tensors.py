import pyMT.data_structures as DS

# listfile = r'E:\phd\Nextcloud\data\Regions\MetalEarth\rouyn\j2\ROUBB.lst' # List file containing sites to write out
listfile = 'E:/my_modules/FFMT/Data/AG/FFMT/allsites.lst'
# data_file = r'E:\phd\Nextcloud\data\Regions\MetalEarth\rouyn\rou_all.dat'
outfile = r'E:\phd\Nextcloud\data\Regions\MetalEarth\rouyn\test_PT.csv'
# outfile = 'E:/phd/Nextcloud/data/ArcMap/PT_test/synth_pt3.csv'   # Outfile file name
verbose = True                  # False only write info needed for ellipses, True write additional PT parameters
scale_factor = 1/50             # Size scale factor for ellipses, measured as a fraction of diagonal window size
                                # e.g., if your stations cover 60 km in X and 60 km in Y (window size of ~85 km),
                                # a scale_factor = 1/50 gives AN ellipse radius of ~1.7 km

data = DS.RawData(listfile)
rmsites = [s for s in data.site_names if s != 'matlab_site83']
data.remove_sites(rmsites)
# data = DS.Data(data_file)
data.write_phase_tensors(out_file=outfile, verbose=verbose, scale_factor=scale_factor)