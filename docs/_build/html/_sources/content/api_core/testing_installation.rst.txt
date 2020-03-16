Testing Installation
====================

Test data is included in the 'test_data' folder.
From that folder, try launching each of the GUIs from the command line:

:ref:`Data Plot`::
 data_plot -n pytest
:ref:`Mesh Designer`::
 mesh_designer testfile_ModEM.dat
:ref:`Model Viewer`::
 model_viewer testfile_ModEM.dat testfile_ModEM.model

If they all launch, you should be good to go.
If not, most likely there is a dependency issue somewhere.