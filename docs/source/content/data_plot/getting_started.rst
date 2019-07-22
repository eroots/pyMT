Getting Started
==========================

Launching the GUI
-----------------

The data plotting GUI is launched from the command line. A number of options exist on exactly how to specify which files you would like to use, which can be accessed using::
	
	data_plot -h

which will output::

	Options include:
         -d : Use default startup file "pystart"
         -b : Browse for start file or data files (Not yet implemented)
         -n : Specify the start file you wish to use
         -l : List the dataset names present in the start file you have chosen
         -c : Choose a specific dataset(s) listed within the start file you have chosen
                 For multiple datasets, separate names with a colon (:)

Options can (and generally should) be used in conjuction.
For example, the most common method of launching the GUI is::
	
	data_plot -n <startup_file> -c <dataset_name>

This will open the :ref:`Startup file` and initialize the GUI with the specified dataset. If no dataset is give (i.e., the -c flag is not used), all datasets within <startup_file> will be loaded.

Required Files
--------------

The data plotting GUI requires at minimum a :ref:`Startup File` containing at least one of the following:

* :ref:`List File`
* :ref:`Data File`
* :ref:`Response File`

Default Behavior
----------------

A few things to note about the default behavior of data_plot:

* Raw data (data read in from EDI / j-format files indicated in a list file) are shown by filled circles
* Inversion data (data read in or created for use in inversion) are indicated by filled circles with a black outline
* Response data (data read in from an inversion response file) is indicated by a solid line

* By default, data is displayed as is. Particularly for impedance data, it is useful to view it multiplied the periods or square root of the periods to be able to visualize both short and long periods equally well. This can be done through the :ref:`Scaling` drop-down menu.

* Some of the features of the Data Plot GUI and the :ref:`Map Viewer` require raw data in order to operate (e.g., addition of periods). Therefore it is generally best to include a :ref:`List File` in all datasets to be plotted.

* The :ref:`Map Viewer` plot is updated any time a contained element is changed. In general this operation is fast. However, if a pseudosection is being plotted in the :ref:`Map Window`, any operation involving a new plot (including changing the viewed sites using the :ref:`Forward and Back Buttons`) may become quite slow.

* When only a :ref:`List File` is specified for the loaded dataset, a :ref:`Data` object will be initialized by taking logarithmically spaced periods.
	* In this instance, the :ref:`Map Viewer` is not initialized properly, and no site locations are shown. The workaround is to re-sort the station locations (e.g., sort by west-east then re-sort back to default) and click the forward or back button. These operations should get the stations plotting.

* When multiple datasets are loaded in, the misfit table may not be properly updated to reflect the currently selected dataset.


.. Known Bugs
.. ----

.. There are some :ref:`Known Bugs` in the data_plot GUI that need to be worked out. In general, these should not break the GUI, but require some workarounds until they are fixed.
