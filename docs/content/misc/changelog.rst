Change Log
==========

*29/04/22

  * Added function to read mt3dani models, and made corresponding changes to allow the anistropic modesl to be used in the GUIs

*01/03/22

	* Forgot to log changes for a while... Not many big ones probably probably many minor ones.

	* Major change is the implementation of site plotting by selection

		* Choose 'Selection' from the 'Sort Sites' dropdown menu. Sites selected in the site list will now be plotted.

		* Not yet thoroughly tested

		* Will eventually want to implement selection by clicking on the map viewer

  * Added plotting of phi_max, phi_min, and beta within the Data Plot window

*13/07/21

  * Removed some of the unused buttons / options from :ref:`Data Plot` and slightly reorganized the layout

  * Added more hover tooltips to the GUIs

  * Added a :ref:`Mesh Designer` button into the Gateway GUI

*01/01/21

  * Added a median filter to 'Regulate Errors' that should help remove outliers so that the generated errors are more sensible

    * Added corresponding spin boxes to :ref:`Data Plot` to control the median filter parameters

* 17/12/20

  * A standalone executable version of pyMT is now available.

    * This is just the GUIs, with the entry point being the new 'Gateway' GUI.

    * No python installation required - the executable contains everything needed.

      * Main drawback is it is only the GUIs - you won't have access to the pyMT API or any of the scripts.

  * Fixed a minor bug where the data cursor (hovering over plots in :ref:`Map Viewer` or :ref:`Model Viewer`) would give slightly incorrect values

  * New 'Gateway' GUI for creating, modifying, and loading your projects (.pymt files, previous 'pystartup' files)

    * Should be compatible with old pystart files, and all previous calling methods should still work (e.g., command line calling of data_plot, model_viewer, etc.)

      * Add a '.pymt' extention to your old project files to get the GUI to automatically recognize them

  * Add Niblett-Bostick depth, PT Skew, and PT Azimuth to pseudosection options in :ref:`Map Viewer`
    
    * Since there are now many options for pseudosections, the radio buttons which previously controlled the plot were replaced with a drop-down menu

  * Added 'XY-YX' to the component option of the pseudosections

    * Only works for App. Rho, Phase, and Bostick depths, but shows XY-YX (TE minus TM) versions

    * Mainly useful for showing phase splitting, but could also be useful for showing variations in penetration depth

  * Similarly, 'Phi_split' has been added to the phase tensor plotting

    * Maybe a misnomer, but this shows XY-YX phases, NOT phi_max - phi_min.

  * Added 'Dimensionality' fill option for PT plotting. This is calculated using skew and ellipticity thresholds of the phase tensor (by default at 3 degrees and 0.1, respectively)

  * Added 'Lambert' projection option in :ref:`Map Viewer`.

  * Added 'Include Outliers' option in pseudosection plotting, along with defineable standard deviation limits

    * Values outside the defined range won't be plotted, so you don't get one station with 400 degree phase colouring your whole plot

  * Data points with the REMOVE_FLAG set (e.g., after using 'Reset Dummy Errors') no longer plot phase tensors in :ref:`Map Viewer`. Should result in much cleaner plots.

* 30/09/20

  * Changed the way missing period fill-in is handled

    * Previously would just grab the closest period - now the data point is 'corrected' for mismatch in period

    * This correction means that the impedance value is different, but the apparent resistivity and phase values are the same

    * Note that this correction is only applied for impedance data - all other data is in-filled as before

  * :ref:`Map Viewer` can now plot tipper amplitude pseudosections

    * Should behave identically to Rho and Phase pseudosections, except that the 'XY / YX/ Det' menu does nothing (only real tipper amplitudes are plotted)

  * Induction arrows and phase tensors now override station location plotting in :ref:`Map Viewer` - no more giant dots hiding PT info!

* 14/08/20

  * Added some logic to 'regulate_errors' so that it would ignore extreme outliers.

  * ModEM has an upper limit on floating point values - exceeding these values causes a crash. Therefore, data and error values are now capped when written to file (and a message is printed).

    * Note that things could still go wrong with this fix (e.g., if data at the cap is inverted, the nRMS is likely to be off the charts). I hesistate to flat out zero such data, and prefer to let the user fix things.

  * Error bars are now shown for average (AAV, GAV, DET) apparent resistivities and phases. The errors are calculated following GEOTOOLS, where the maximum error between XY and YX is taken.

  * Fixed a bug where :ref:`Map Viewer` would let you try to change the coordinate system even if no raw data was loaded, and promptly crash. 

  * Fixed a few issues with isosurface plotting in :ref:`Model Viewer`

    * Isosurface will now automatically refresh when recalculated

    * Added opacity toggles so that you can actually see multiple isosurfaces if plotted

    * Added try/except block to make sure you can't plot a contour that has no values (e.g., if your desired contour line is lower than all the values in the model)

  * Added toggle to plot station locations at their inverted elevation.

  * Added some canned background colours for the 3D view in :ref:`Model Viewer`:. Mostly cosmetic, but also useful if you are plotting transparency based on resolution.

* 04/08/20

  * The :ref:`Map Viewer` 'Lock Axis' option should now properly hold the axis limits when site annotations change (i.e., when using the forward and back buttons in :ref:`Data Plot`)

  * Periods (and data) will now be sorted in ascending order at the Site (class) level.

    * Having some EDIs with periods in ascending and some in descending was causing data at different sites to have different orders.

    * I don't think this was having any significant effects (and so this fix shouldn't change anything), but it was still worrying.

* 22/07/20
  
  * Added Complex Apparent Resistivity Tensor (CART) representation into :ref:`Map Viewer`

    * A drop down menu in :ref:`Map Viewer` can be used to switch between conventional PT and CART ellipses

    * Note that not all of the ellipse fill values will be meaningful when display CART ellipses.

      * Most useful parameters will be 'Phi_max' and 'Phi_min'. Note that this always correspond to the maximum and minimum axis values, be it phase (in the case of PT and RPT) or resistivity (in the case of Ua and Va)

    * Also note that while I have tested and compared the plotted CART ellipses against identical data plotted using FFMT (Frankfurt MT Software package, where CARTs were created), this feature is still experimental, and there is a possibility that some features supported by the conventional PT plotting tools have not been properly applied to the CARTs.

  * Linked more plot elements to the colour scales that can be set in :ref:`Map Viewer`

    * Rho pseudosections, model slices are controlled using the 'Rho' colour limits. Real resistivity tensor phi_max / phi_min values will be coloured by a log scale colour bar going from -U, U, where U is the upper 'Rho' colour limit. Imaginary resistivity tensor will use a linear scale from -U, U.

    * Phase pseudosection and non-rotational phase tensor parameters (e.g., det_phi, phi_min, etc.) will use the 'Phase' colour limits. Resistivity phase tensor will use -U, U, where U is the upper 'Phase' colour limit.

* 05/07/20

  * Added options in :ref:`Map Viewer` to change the rotation axis definition for phase tensors

    * By default it was (is) X-axis, meaning alpha, beta, azimuth are calculated counter-clockwise from X

    * Alternate definition is to measure clockwise from Y

    * Note that this only changes the numerical values and therefore the colours alpha, beta, and azimuth, but not the orientations.

  * Added some logic in the IO module to allow slightly more robust reading of EDI files

    * Locations where only being read from the 'DEFINEMEAS' block, but will now pull from 'HEAD' if the former is not defined.

  * Added 1D modelling to :ref:`Data Plot`

    * Open another window that allows you to enter layer thicknesses and resistivities, and the calculated response can then be plotted across all your stations

    * Meant to allow for quick comparision between your data and the response for a 1D model.

    * TODO: Allow writing of the 1D model.

* 25/05/20

  * Fixed a bug that were causing 'Azimuth' and 'Alpha' to be displayed improperly (colours only, PT orientations were always fine)

    * This bug fix should also fix issues with exported phase tensors in ArcMap not matching those plotted with pyMT

  * Fixed bug which caused a 'transect slice' in :ref:`Model Viewer` to use technically out-of-bounds locations, and therefore use a fill value instead of the actual model values.

* 14/05/20

  * Cleaned up a few things that would crash :ref:`Data Plot` (e.g., checking boxes that should be uncheckable)

  * Added some new colour options

    * You can now control LUT (number of colour intervals). I realized that while 16 or 32 is good for viewing models, it might remove necessary details when viewing things like phase tensors

    * New cyclic colour maps 'twilight' and 'colorwheel' added. Useful for viewing wrapped quantities like phase tensor azimuth.

    * Removed second 'Colour Options' menu in :ref:`Map Viewer` and consolidated those options into one menu. All colour map / limits / LUTs are now controlled in that one menu.

    * Fixed and issue where model slice colour map was not responding to changes in the colour limits

* 28/04/20

  * A few QoL changes in :ref:`Data Plot`:

    * The error tree will now properly collapse and expand nodes when you flip through the stations.

    * Fixed a bug where removed sites were still being considered when plotting induction arrows, PTs, and pseudosections in :ref:`Map Viewer`

    * Added controls for data period tolerances (in the :ref:`Error Manipulations` tab)

      * 'Flag' tolerance sets selected periods without a cooresponding period in the Raw Data within said tolerance to have increased errors.

      * 'Remove' tolerance sets periods outside said tolerance to be flagged for removal. Flagged data points are placed at the end of the ModEM data block, with errors of 1234567. Use your favourite text editor to remove the block.

    * Correspondingly, controls were added to remove these points from the plots. Note that the plots in :ref:`Map Viewer` will still include the flagged data points.

    * If you attempt to write a ModEM file with flagged data, you will be asked if you want to write out 2 versions of the data file (one with the flagged data, one without). The version without will have '_remove' appended to your output file name.

* 23/04/20

  * Changed behavior of transect plotting in :ref:`Model Viewer` such that it automatically plots and refocuses the GUI into the 3D view.

  * Changed sizing policy of various :ref:`Model Viewer` components to hopefully eliminate some of the window resizing bugs.

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