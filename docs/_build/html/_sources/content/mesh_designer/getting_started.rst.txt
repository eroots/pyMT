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

If only a data file is used, an initial model will be created based on the bounds of the given stations. There is a known bug here where the inital view of the model cuts of the outer edges. Hitting the 'Add Pads' a few times will extend the model out a bit to cover the whole area covered by the stations.

If both model and data files are given, the mesh will be shown as it is in the given model, with the station locations overlaid.

As this GUI is not meant for viewing the model, the slice shown is always the first depth slice. 

The GUI works equally well for creating new meshes as it does for modifying existing ones.

* Note: As the definition of the mesh within the ModEM and WSINV3DMT file formats has no explicit origin, all models generated here will have their origins placed in the center of the mesh.
* For this reason, it is important to ensure that the model is even on the left / right and top / bottom. 
	* If it is not, the definition of the mesh relative to the station locations may not be the same as appears while using this GUI.
* Always double check the output mesh and data files with another tool afterwards (e.g., with :ref:`Model Viewer`


