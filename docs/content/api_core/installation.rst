Installation
============

Dependencies
------------

* Python 3.5 (or greater)
* Numpy
* SciPy
* matplotlib
* pyqt
* colorcet
* pyshp
* pyproj
* pyvista
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