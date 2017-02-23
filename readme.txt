'pyMT' python project.
Base Modules include:
	data_structures: Contains classes used for storing and manipulating MT inversion data and models.
	IO: Module containing functions related to reading and writing wsinv3dmt related files. Most functions can be accessed through the relevant classes in the data_structures module.
	utils: Contains utility functions used by the various data_structure classes and data plotters contained in this project.
	gplot: A data manager for plotting data. Used for more advanced / customizable plotting, but best accessed through data_plot.py (in DataGUI directory)
	
Submodules:
	DataGUI: Contains modules used to interactively plot and manipulate
		 wsinv3dmt data and models. Not yet set up for interactive file browsing. GUI is launched with by calling it through python, i.e. 'python data_plot.py', and the relevant files are read through startup files. See data_plot.py for more information.
	tests: Contains modules used to test the various parts of the project