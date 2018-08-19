import simplekml
import pyMT.data_structures as WSDS

# list_file = r'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/wst/New/j2/all.lst'
# list_file = r'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/abi-gren/New/j2/allsites.lst'
# list_file = r'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/MetalEarth/dryden/j2/allsites.lst'
list_file = r'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/MetalEarth/rainy/j2/allsites.lst'
save_path = r'C:/Users/eric/phd/ownCloud/Metal Earth/Data'
save_file = '/'.join([save_path, 'rainy.kml'])
raw = WSDS.RawData(listfile=list_file)

# raw.locations = raw.get_locs(mode='latlong')
kml = simplekml.Kml()
for site in raw.site_names:
    lat, lon, elev = (raw.sites[site].locations['Lat'],
                      raw.sites[site].locations['Long'],
                      raw.sites[site].locations['elev'])
    kml.newpoint(name=site, coords=[(lon, lat, elev)])

kml.save(save_file)
