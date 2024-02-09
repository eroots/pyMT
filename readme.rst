pyMT
====

Codebase for visualization, analysis, and preparation of magnetotelluric data for inversion (compatible with ModEM, MT3DANI, and WSINV3DMT)

Installation basics (tested method for new users)

1) Install anaconda (or miniconda; skip this if you already have anaconda)

2) Open anaconda prompt

3) Create an environment for pyMT, re-load a few dependencies::

	conda create -n pymt python conda pip git setuptools

4) Enter that environment::

	conda activate pymt

5) Add that command to your .bashrc (or whatever login script) so you don't have to enter it every time you open a new anaconda prompt

6) Install pyMT::
	
	pip install git+https://github.com/eroots/pyMT.git

7) Any time you want to update to the latest version, use::

	pip install --upgrade  git+https://github.com/eroots/pyMT.git

8) Navigate to wherever your working folder (typically the highest folder containing your MT data and workflow) is and launch::

	gateway_mt

Note: The dependence on natural-neighbor has been removed to improve the installation process. The default interpolation method for plan-view pseudosections is now based on the linear kernel of the scipy radial basis function interpolator (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html).

If you prefer the natural neighbor interpolation, an updated version (compatible with the latest versions of python and numpy) can be installed via::

	pip install git+https://github.com/eroots/natural-neighbor-interpolation

Note that the natural-neighbor package requires Microsoft Visual C++ Redistributables. These may already be installed, but on newer PCs you may have to first install Microsoft Visual Studio Build Tools (https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)

Getting Started:

All of the GUIs (data_plot, mesh_designer, model_viewer) can now be accessed through one main window, which is launched using the command::

		gateway_mt

This GUI is essentially just a table containing all the files needed for a series of datasets (where a dataset is any permutation of a list file, and inversion data/response/model files). Highlight a dataset and click the 'Data Plot', 'Mesh Designer', or 'Model Viewer' buttons to launch the relevant GUI with the highlighted dataset.


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

