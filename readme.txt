'wsinv3dmt' python project.
Base Modules include:
	data_structures: Contains classes used for storing and manipulating MT inversion data and models.
	IO: Module containing functions related to reading and writing wsinv3dmt related files. Most functions can be accessed through the relevant classes in the data_structures module.
	utils: Contains utility functions used by the various data_structure classes and data plotters contained in this project.
	
Submodules:
	DataGUI: Contains modules used to interactively plot and manipulate
		 wsinv3dmt data and models
	tests: Contains modules used to test the various parts of the project

== CHANGELOG ==
25/01/17 - Fixed bug where dataset would print error about unequal
24/01/17 - Print periods in data_plot.py now prints fixed width columns with headers
22/01/17 - Refactored command line parsing out of main() from data_plot.py.

