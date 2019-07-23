Input File Types
================

.. _Startup File:

Startup File
************

The startup file specifies the relevant paths and files, as well as the roles of those files. 
The startup file is broken into seperate data sets, with a '%' denoting the beginning of a new data set specification, and the subsequent lines giving the role and paths of the files to be used. Lines starting with a '#' will be ignored, which allows the insertion of comments.
A data set can contain one `List File`_, `Data File`_, and `Response File`_, or any permutation thereof (as long as only one of each is specified per data set). Additionally, you can specify a common path to each file, as well as a separate path which points to the location of the raw data files (EDI or j-format files), although this has not been fully tested, and so specifying relative or absolute paths is the safer option for now.
An example startup file is as follows::

	# This specification assumes allsites.lst, inv.dat, and the EDI/j-format files are all in the same folder as pystart
	% data_set1  
	list allsites.lst 
	data inv.dat
	# The list and EDI files are in a folder ./EDIs
	% data_set2
	list EDIs/broadband.lst
	# The list file and EDIs are in ./EDIs and the data and response files are in ./inversion2
	% data_set3
	list EDIs/broadband.lst
	data inversion2/broadband.dat
	resp inversion2/inv_response.dat

Assuming this file is called 'pystart' and the terminal is in the same folder, the GUI can be launched with (for example)::

	data_plot -n pystart -c data_set1:data_set2

.. _Data File:

Data File
*********

The data files are those that are used as input to your inversions.
Currently implemented formats are:

* ModEM (2-D)
* ModEM (3-D)
* WSINV3DMT
* Occam2D
* MARE2DEM

Thus far, only ModEM and WSINV3DMT file formats have been used extensively. Other formats, while implemented, have not been thoroughly tested.

Data file handling is implemented through the :ref:`Data` class.

.. _Response File:

Response File
*************

The response file output from an inversion. If the format of data and response files is the same for your given inversion code (as is the case for ModEM), then data and response files may be used interchangeably.

Currently implemented formats are the same as for the `Data File`_
Response file handling is implemented through the :ref:`Response` class, which is largely just a subclass of :ref:`Data`.

.. _List File:

List File
*********

A list file specifies the EDI or j-format files you would like to import.
The first line specifies the number of stations contained in the file, and each subsequent line is the name of a station. The names can specify .EDI or .dat to specifically read in EDI or j-format files. The file-reader will look for both if no format is specified, preferentially selecting j-format files if both formats are present.

List files can be used to assign station names to when read into a :ref:`Data` object (for instance, a normal WSINV3DMT data file does not contain station names), or to specify the files to be read into a :ref:`RawData` object.

.. _Model File:

Model File
**********

The model files that are used as input to and output from the inversions.
Currently implemented formats are:

* ModEM (2-D)
* ModEM (3-D)
* WSINV3DMT

Thus far, only ModEM and WSINV3DMT file formats have been used extensively. Other formats, while implemented, have not been thoroughly tested. Furthermore, 2-D ModEM model files have not been used much, and so may result in some unpredictable behavior.

Model file handling is implemented through the :ref:`Model` class.