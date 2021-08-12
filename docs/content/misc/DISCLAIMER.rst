DISCLAIMER
==========

This code base has been cobbled together over a few years. Parts of it were written to fulfill a specific need at a specific time and then promptly forgotten, while others were written when I was still figuring out the difference between a class and a method. While I have tried to address bugs as I come across them, this code is not unit tested (yet), and so is best used with an understanding of the the expected outcome is. There are some `known bugs`_, but any further unexpected behavior can be reported.


.. _Known Bugs:

KNOWN BUGS
==========

General
-------

* Text is occasionally printed to the terminal. This text is usually meant to convey some information about something unexpected, and the codes attempts to work around it. Occasionally text will be printed that was meant for debugging purposes, and has just not been removed. Generally, any text that is printed that isn't followed by a crash is fine.


Data Plot GUI
-------------

* When only a :ref:`List File` is specified for the loaded dataset, a :ref:`Data` object will be initialized by taking logarithmically spaced periods.
	* In this instance, the :ref:`Map Viewer` is not initialized properly, and no site locations are shown. 
	* The workaround is to re-sort the station locations (e.g., sort by west-east then re-sort back to default) and click the forward or back button. These operations should get the stations plotting.

* Rotating the data / stations using the 'Azimuth' box has a few associated bugs
	* The station locations and annotations in the :ref:`Map Viewer` may not be accurate
        * Non-zero azimuths also do something with the ordering of the stations in :ref:`Map Viewer`, and cycling through stations with the :ref:`Forward and Back Buttons` changes the order on the map. Will try to fix this.
	* The Azimuth editor is meant to be used for 3-D data, and rotates the data accordingly: Station locations are rotated clockwise from north, and the data is rotated counter-clockwise to maintain a measurement coordinate system that is consistent with the model space (see https://www.linkedin.com/pulse/grid-sites-data-rotations-3d-mt-dr-naser-meqbel/)
		* A consequence of this setup is that the GUI is not suitable to rotation and projection into 2-D.

* The 'Write All Plots' action sometimes crashes the GUI.
	* Known instances of this are when you attempt to overwrite an open PDF file. A permission error is thrown and not caught, resulting in a crash
	* Occasionally, if the number of subplots in the final saved plot is different from that specified within the 'Plot Options' tab, the next action taken can result in a crash.

* If more subplots are used than there are stations available, one station will be re-plotted in order to fill the unused subplots.

* Blank items in the :ref:`Data Component Table` are selectable. However, this results in the first available component being plotted instead, and so should not break the GUI.

* The RMS misfits as shown in the :ref:`Misfit Table` may differ slightly from what logged by ModEM as pyMT always applies an error floor to any data read in. ModEM does not explicitly use (or store) an applied error floor, and so the hard-coded error floors of pyMT (see :ref:`Error Floors`) may differ from those used in the inversion. As a result, if you used a lower error floor than those coded into pyMT, the misfits shown in this table will be lower.

Model Viewer GUI
----------------

* Hovering over the plots within the 2-D will show location and resistivity information about the cursors position. This seem to generally be correct, however sometimes the resistivity shown does not match the plot itself. Likely an issue with matching the cursors location to model cells near the edges of the model.

* The 2-D transect plot shows left-right in the order that was clicked.
	* This means that if you select points from north to south, the figure will plot from north on the left to south on the right.
* The 2-D transect plot does not respond to changes in the model trim. The workaround currently is to trim the volume, and then re-select the desired points for the transect plot.

