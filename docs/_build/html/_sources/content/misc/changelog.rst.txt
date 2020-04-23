Change Log
==========
* 23/04/20

  * Changed behavior of transect plotting in :ref:`Model Viewer` such that it automatically plots and refocuses the GUI into the 3D view.

* 03/04/20

  * Changed 'Lock Axis' behavior in :ref:`Data Plot` to lock bounds to static values, defined in the 'Display Options' menu.

* 30/03/20

  * Removed dependency on naturalneighbor. :ref:`Data Plot` will now offer other options for interpolation. If naturalneighbor happens to be installed, this option will appear.

* 14/03/20
  
  * Fixed a few of the issues related to reading multiple data sets into Data Plot
    
    * Use the 'Recalculate RMS' button in the :ref:`Data Selection` tab to refresh the :ref:`Misfit Table` after switching the data set.
  
  * Inversion type is detected from available components when loading only a list file into :ref:`Data Plot`
    
    * This 'should' fix the bugs related to :ref:`Map Viewer` not allowing plotting of induction arrows and phase tensors.
  
  * Fixed bug which stopped station locations from being plotted when using only a list file.
  
  * Added 'Coordinate System' in the :ref:`Map Viewer`. Stations can be plotted in local, UTM, or lat/long. Note that which of these is available will depend on what data is loaded (e.g., a ModEM data file alone has no information about the geographic locations of the stations)
  
  * Added a 'JPEG' menu in :ref:`Map Viewer`. This allows loading of a geo-referenced JPEG image into the background. So far I have only tested it with UTM referenced JPEGs (and its corresponding world file), but I don't see any reason why a lat/long referenced file wouldn't work.
    
    * Note that when plotting these background images, the Coordinate System needs to be set appropriately.
  
  * Added some documentation in :ref:`Recipes` outlining my approach to working with data with non-uniform periods and / or components.

* 01/02/20
  
  * Models can now be read into 'pystart' files in :ref:`Data Plot`.
    
    * If loaded, plan view slices can be plotted in :ref:`Map View`.

* 26/01/20
  
  * Can now update the RMS table with a button after changing the plotted dataset.
  
  * Plotting of imaginary tipper arrows.
  
  * Added a legend for induction arrows showing colours and reference lengths

* 23/12/19
  
  * Added option to set equal or auto aspect ratio in the :ref:`Map Window` of :ref:`Data Plot`
  
  * Also added freezing of axis limits, so you can zoom / pan and keep the same view after changing what is plotted.

* 14/12/19
  
  * Fixed induction arrow plotting in data_plot so that un-normalized arrows are actually useable. Not thoroughly tested however.
  
  * Added option to specify a 'cutoff' length for induction arrows. Arrows with magnitudes greater than this will not be plotted.
  
  * Added secondary phase tensor plotting as inner bars within the phase tensor ellipses (as in Hering et al., 2019)
  
  * Fixed bath2model script to properly specify ocean and air cells within the covariance file.
  
  * In the process of fixing and testing how covariance files need to be written.

* 28/11/19
  
  * Fixed a bug where ModEM data files would include elevations if data was read directly from EDI files (which would put the receivers underground)
  
  * Added a static value to the Data class 'Data.REMOVE_FLAG', which is meant to be assigned to data points you want removed from the inversion data file
    
    * Currently not functional with the GUIs, but can be used to assign recognizable error values to data points to be removed, which can then be removed manually
    
    * Will (eventually) add these things into the GUIs...
      
      * For now, see :ref:`Recipes` for an example on how to assign the errors and remove the data points from a ModEM data file.
  
  * Added an option to write model to CSV file (accessible via the API only right now)
  
  * Added functionality to read / write 2-D ModEM models and data
    
    * Still buggy and less than ideal. Is you're data really that 2-D anyways?

* 10/11/19
  
  * ModEM data file read function now checks for sign convention and units
    
    * Will automatically convert to exp(-iωt) and ohms

* 01/10/19
  
  * Added a script to add oceans and topography
    
    * Still experimental - model seems to be built correctly, but covariance file needs corrections
  
  * To fit above, changed default behavior of data file writing:
    
    * By default, elevations will not be used (i.e., Z = 0 for all stations). Add 'use_elevation=True' as a named parameter in your write to include elevations
    
    * Note that due to a bug, previous versions may have included elevations in the written ModEM data files which could lead to spurious results.

* 09/09/19
  
  * Need to add these to the docs changelog when I get that fixed
  
  * Added turbo and turbo_r to colour maps
      
      * Should automaticaly be working in Model Viewer, not yet in Data Plot
  
  * Changed the way ModEM data files are read in to allow for arbitrary ordering of the data lines
      
      * This seems to be working without complaint, but may have some unintended side effects!

* 08/08/19
	
  * Some changes to IO to start to allow different periods for different sites (not fully implemented yet)
	
  * Some bug fixes related to reading data files

* 30/07/19
	
  * :ref:`Model` class can now read and write model covariance files
	
  * :ref:`Mesh Designer` will automatically prompt for covariance file output when writing a model.
	
  * Added documentation for some of the more usuable scripts.

* 23/07/19
	
  * Re-release of pyMT onto GitHub
	
  * Now with (some) documentation!
		
    * See the pyMT/docs folder for a PDF version, or pyMT/docs/build/html/index.html to load up a browser version (complete with navigation bar and search tool)