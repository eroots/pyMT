'pyMT' python project.
Base Modules include:
	data_structures: Contains classes used for storing and manipulating MT inversion data and models.
	IO: Module containing functions related to reading and writing wsinv3dmt related files. Most functions can be accessed through the relevant classes in the data_structures module.
	utils: Contains utility functions used by the various data_structure classes and data plotters contained in this project.
	
Submodules:
	DataGUI: Contains modules used to interactively plot and manipulate
		 wsinv3dmt data and models
	tests: Contains modules used to test the various parts of the project

== CHANGELOG ==
10/02/17 - Added a few tools to utils, including a filtering operation that can be used, for example, to smooth rho or phase prior to bostick transform.
	 - Added options for geometric / arithmetic average (gav/aav) calculations to the compute_rho/phase functions in utils.
	 - Added compute_bost1D to utils, which calculates the 1-D Bostick-Niblett transform for a given site.
	 - RawData objects now reject dummy period data when data is read. Data whose values are constant across all periods are removed. This is checked via the method check_dummy_periods, and if any sites are found with dummy data, the periods are removed with remove_periods
07/02/17 - Fixed a bug where Data.remove_site() would remove sites from the dictionary and site list, but not data.locations
	 - Changed the precision of the data printed in IO.write_data to match that printed out using j2ws3d
	 - Stopped tracking the various pystart files in DataGUI/
27/01/17 - Added TODO.txt, changed remote to Github repo rather than having working folder --> local repo --> local bare repo --> remote repo
25/01/17 - Renamed wsinv3dmt module to pyMT, and made relevant changes to the submodules
	 - Fixed bug where dataset didn't actually fix unequal azimuths
	 - Renamed main module from wsinv3dmt to pyMT, made changes to submodules to reflect this.
24/01/17 - Print periods in data_plot.py now prints fixed width columns with headers
22/01/17 - Refactored command line parsing out of main() from data_plot.py.

