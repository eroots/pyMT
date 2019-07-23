Getting Started
==========================

Launching the GUI
-----------------

The model viewer GUI is launched from the command line via the command::
	
	model_viewer <model_file> <data_file>

The order of the files does not matter.

It can take a while to load the GUI, particularly the first time you do so, due to the somewhat heavy dependencies required for 3-D viewing.

Required Files
--------------

The model plotting GUI requires at minimum a :ref:`Model File`, and an optional :ref:`Data File`, which is currently only used to plot station locations.

Default Behavior
----------------

Assuming valid files are used, the GUI should launch into the 2-D view, with the slice locations set to X=1, Y=1, Z=1 (bottom, left, and surface slices, respectively)

The 3-D view is initialized to a top-down (XY) view.

Default colour map is 'jet_plus', a modified version of the Matlab default 'jet' with lower and upper colour map limits of 1 and 5 (log10 scale).

Note that currently there is no colour bar shown for the 2-D views.

Currently the data file is used only to plot the station locations over the model. Plotting of induction arrows and phase tensors will likely be added in the future.


