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

Install the more complicated prerequisites::

	conda install numpy
	conda install vtk

Install by navigating to the cloned pyMT directory and entering::

	python setup.py install

Or if you intend to modify the code::

	python setup.py develop

It is recommended you use the 'develop' option regardless at this time.

Most dependencies will be installed during this process, however if you are coming from a clean python installation, it may be best to manually install certain packages with potentially complicated dependencies.
These include:

* pip
* numpy
* scipy
* matplotlib
* colorcet
* vtk

The tested method of installation involves installing each of these with Anaconda prior to installing pyMT. As of the last time I tested this (December 2020), the only ones that needed to be manually installed were vtk and numpy.

Note, a previously required package 'naturalneighbor' has been removed from the dependency list. :ref:`Data Plot` will now instead offer other interpolation schemes based on scipy. If naturalneighbor is installed (it can be installed through pip, or using the wheel at https://www.lfd.uci.edu/~gohlke/pythonlibs/#naturalneighbor), the natural neighbor scheme will still be available.

Other dependency issues:

 * Python needs to be version 3.7.*
 * PyQT5 needs to be version >5.14.*

An update to pyvista version 0.24.* has also caused an error in setting up the :ref:`Model Viewer`, so pyvista version 0.23.* is required for now. This requirement is included in the setup.