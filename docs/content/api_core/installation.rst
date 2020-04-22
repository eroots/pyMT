Installation
============

Dependencies
------------

* Python=3.7.*
* Numpy
* SciPy
* matplotlib
* pyqt5=5.14.*
* colorcet
* pyshp
* pyproj
* pyvista=0.23.*
* pyvtk
* naturalneighbor

Installing pyMT
---------------

Clone the repository::

	git clone https://github.com/eroots/pyMT.git

Install by navigating to the cloned pyMT directory and entering::

	python setup.py install

Or if you intend to modify the code::

	python setup.py develop

All dependencies will be installed during this process, however if you are coming from a clean python installation, it may be best to manually install certain packages with potentially complicated dependencies.
These include:

* pip
* numpy
* scipy
* matplotlib
* pyqt
* colorcet

The tested method of installation involves installing each of these with Anaconda prior to installing pyMT.

Note, a previously required package 'naturalneighbor' has been removed from the dependency list. :ref:`Data Plot` will now instead offer other interpolation schemes based on scipy. If naturalneighbor is installed (it can be installed through pip, or using the wheel at https://www.lfd.uci.edu/~gohlke/pythonlibs/#naturalneighbor), the natural neighbor scheme will still be available.

As of the time of writing this (April 2020), there is an issue between versions of python and pyqt that may cause some GUIs to fail:

 * Python needs to be version 3.7.*
 * PyQT5 needs to be version 5.14.*

Note that as of now, Anaconda only seems to have access to pyqt5 version 5.9.*. You will therefore have to install pyqt5 using pip, and ensure this version supercedes any other version of pyqt installed in your environment.

An update to pyvista version 0.24.* has also caused an error in setting up the :ref:`Model Viewer`, so pyvista version 0.23.* is required for now.