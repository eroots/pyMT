Getting Started
==========================

Launching the GUI
-----------------

The mesh designer GUI is launched from the command line via the command::
	
	mesh_designer <model_file> <data_file>

The order of the files does not matter.

Required Files
--------------

The model plotting GUI requires one or both of the following:
* :ref:`Model File`
* :ref:`Data File`


Default Behavior
----------------

The default behavior is different depending on what files are input.
If only a model file is used, the GUI initialized using the given model, and no station locations are plotted. In this case, the 'Regenerate Mesh' button will not be functional.

If only a data file is used, an initial model will be created based on the bounds of the given stations. There is a known bug here where the inital view of the model cuts of the outer edges. Hitting the 'Add Pads' button will reset the view to correctly display the mesh.

If both model and data files are given, the mesh will be shown as it is in the given model, with the station locations overlaid.

As this GUI is not meant for viewing the model, the slice shown is always the first depth slice. 

The mesh lines are white, and therefore may be difficult to see over some models depending on the resistivity. Other options for mesh line colour will be added in the future, but for now the workaround is to change the colour map or colour limits to allow the white lines to be seen clearly.

