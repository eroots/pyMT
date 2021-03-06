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
	ModelGUI: Contains the modules and UI's for creating model meshes. Currently supports 3-D models for ModEM and WSINV3DMT

Note: This package does not automatically install dependencies. You will need to manually install them (Some may come automatically with newer versions of Anaconda). Some may require you to specify where to download it from. If the default 'conda install ___' doesn't work, do a quick Google search.
Dependencies include:
	matplotlib
	numpy
	pyproj
	pyshp
	pyqt5
	e_colours (Ask me to about getting this one. If it becomes a problem, I can include it directly in the package)
	pyvista *NEW*
	vtk	*NEW*