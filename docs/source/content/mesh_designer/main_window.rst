Mesh Designer - Main Window
===========================

.. figure:: ../../images/mesh_designer.png
    :align: center
    :scale: 50 %

The Mesh Designer has 3 areas: The `Menu Bar`_, `Control Dock`_, and `Plot Window`_.

Menu Bar
--------

.. figure:: ../../images/mesh_designer_menu_bar.png
    :align: center
    :scale: 50 %

The Save / Revert menu has options for writing the model to a file, as well as saving and reverting progess within the Mesh Designer.

The Write Model sub-menu will open a dialog box that asks for the name of the file to be written to. You will be prompted again if this operation is going to overwrite an existing file.

* The only file format available from here is that of a ModEM 3-D file. This is the same as the WSINV3DMT file format, except that the resistivity values will be given as the natural logarithm.
 
* If you need the file in WSINV3DMT format, the workaround at present is to change the file format programmatically (see :ref:`Recipes`)

The Save Progress sub-menu internally saves any changes you've made to the mesh. This checkpoint can then be reverted to later by using the Revert Progress sub-menu.

The Colours menu is controls the colour map and colour limits used, as well as the colour of the mesh lines.

* As this GUI is meant mainly for editing the mesh, these options are mainly to ensure good visibility of the mesh lines regardless of the resistivity used

Control Dock
------------

.. figure:: ../../images/mesh_designer_control_dock.png
    :align: center
    :scale: 50 %

The Control Dock is the main control panel for manipulating the mesh and model. It is broken into 3 tabs:

* `Manipulate Mesh`_
* `Reset Mesh`_
* `Smoothing`_

Manipulate Mesh
^^^^^^^^^^^^^^^

The Manipulate Mesh tab is used to, as the name suggests, to modify and manipulate the mesh used.

Specifically, there are 3 things that can be modified from this tab: The XY Padding, the Z mesh (or depth mesh), and the background resistivity.

The XY Padding groupbox is used to add and remove padding cells in the XY plane.

The Add Pad and Remove Pad buttons add and remove pads from the left / right / top bottom/ of the XY plane, depending on which of the corresponding checkboxes are selected.

When adding pads, the size of the new pad is determined by taking the size of the outer most cell and multiplying it by the value in the Pad Size Multiplier spinbox.

* Note: As the definition of the mesh within the ModEM and WSINV3DMT file formats has no explicit origin, all models generated here will have their origins placed in the center of the mesh.
* For this reason, it is important to ensure that the model is even on the left / right and top / bottom. 
	* If it is not, the definition of the mesh relative to the station locations may not be the same as appears while using this GUI.
* Always double check the output mesh and data files with another tool afterwards (e.g., with :ref:`Model Viewer`)

Z Mesh
^^^^^^