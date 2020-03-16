Scripts
=======

A number of scripts come with pyMT, although only a few of these are ready for `General Use`_ . Others will need to be manually edited for your particular needs. Only a few will be described here.

General Use
-----------

The scripts listed here are ready to run out of the box, and either have command line inputs to control them, or are so simple as to not require any inputs.

j2ws3d
^^^^^^

j2ws3d is a command line tool for preparing data for inversion. It is included in your search path upon installtion of pyMT, so it can be run from anywhere (preferably wherever your raw data is).

Note, that by default j2ws3d.py tries to create both models and data through command line inputs, but the model creation aspect is bugged and will crash. Either run::
	
	j2ws3d.py -data

to initial a data only mode, or just ignore the program crash (as it occurs after the data file is written out).

Other than that, just follow the on-screen prompts.

This program has is still usuable, but has been superceded by the :ref:`Data Plot` GUI.

to_vtk
^^^^^^

Script for converting site locations and models to VTK files (compatible with, e.g., Paraview).

Usage::

	to_vtk.py

Follow the command line prompts to select your model and/or data files. You will also have the option to project your model and data into a particular UTM zone.

Note that any station / model rotations will have to be handled manually.

ws2modem
^^^^^^^^

Simple script to convert WSINV3DMT data files into ModEM 3-D data files.
Usage::

	ws2modem.py <ws_input_data> <modem_output_data>

