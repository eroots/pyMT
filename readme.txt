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

27/01/17 - Added TODO.txt, changed remote to Github repo rather than having working folder --> local repo --> local bare repo --> remote repo
25/01/17 - Renamed wsinv3dmt module to pyMT, and made relevant changes to the submodules
	 - Fixed bug where dataset didn't actually fix unequal azimuths
	 - Renamed main module from wsinv3dmt to pyMT, made changes to submodules to reflect this.
24/01/17 - Print periods in data_plot.py now prints fixed width columns with headers
22/01/17 - Refactored command line parsing out of main() from data_plot.py.

