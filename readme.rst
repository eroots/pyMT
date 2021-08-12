pyMT
====

Codebase for visualization, analysis, and prepartion of magnetotelluric data

Base Modules include:
	data_structures: Contains classes used for storing and manipulating MT data and models.

	IO: Module containing functions related to reading and writing files.

	utils: Contains utility functions used by the various data_structure classes and data 
	plotters contained in this project.

	gplot: A data manager for plotting data. Used for more advanced / customizable plotting, but largely used by the included GUIs
	
Submodules:
	DataGUI - Contains modules used to interactively plot and manipulate data and models.

	ModelGUI - Contains the modules and UI's for creating and viewing model. Currently supports 3-D models for ModEM and WSINV3DMT

	tests - Contains modules used to test the various parts of the project. Not up to date.

Installation:
	All dependencies will be installed during this process, however if you are coming from a clean python installation, it may be best to manually install certain packages with complicated dependencies.
	These include:

	* pip
	* numpy
	* scipy
	* matplotlib
	* pyqt
	* colorcet
	
	Each of these can be installed manually using, e.g., conda install pip (assuming an Anaconda installation)
	
	After installing these, pyMT may be installed using the command 'python setup.py install' or 'python setup.py develop'

Getting Started:
	All of the GUIs (data_plot, mesh_designer, model_viewer) can now be accessed through one main window, which is launched using the command::

		gateway_mt

	This GUI is essentially just a table containing all the files needed for a series of datasets (where a dataset is any permutation of a list file, and inversion data/response/model files). Highlight a dataset and click the 'Data Plot', 'Mesh Designer', or 'Model Viewer' buttons to launch the relevant GUI with the highlighted dataset.
