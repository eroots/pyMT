For most users pyMT is best accessed through its GUIs.
Right now, those are:
	data_plot
	mesh_designer
	model_viewer

data_plot
	- Displays MT data read in from EDI, ModEM, WSINV3DMT, MARE2DEM, Occam2D, and j-format files
	- Running it:
		- requires a startup file (I'll call it 'pytest') that contains the files you wish to include
			- Template of this is:
				% dataset_1       # Dataset name. Will be used when calling data_plot
				list list1.lst    # file with the names of the EDI or j-format files to read
				data data1.dat    # Data to be used as input to inversion
				resp resp1.dat    # Output response file from inversion
				% dataset_2
				list list2.list
				data data1.dat
				...
				...
				etc.
			- Note any combination and order of list data and resp can be written, as long as at least 1 is present
			- The .lst file contains the number of stations in the 1st line, then each subsequent line is a station name.
		- GUI is run by entering (from within the folder with the startup file)
			data_plot -n pytest -c dataset_1
		where pytest is your startup file, and dataset_1 is the dataset within that file you wish to view.
		- You can also run data_plot -n pytest -l (lowercase L) to see all datasets within a startup file, or data_plot -n pytest -c dataset_1:dataset_2 to load multiple datasets in
		- Enter data_plot -h to get a list of these and more options
		- Once the GUI is running, you can view any of the data components that were included in the files you choose
			- Note that only those components common to all files will be shown (e.g., if the EDIs contain tipper, but the inversion data file does not, then no tipper will be shown)
			- Rho and Phase will be available calculated for the available components
			- Note that when viewing the impedance data, it is generally useful to view it scaled by sqrt(period), which is accessed through the 'Scaling' menu within the 'Plot Options' tab
			- Change the currently viewed stations with the 'Forward' and 'Back' buttons
		- You can add or remove periods from the inversion as follows:
			- Click the 'Add Periods to Data' checkbox in the 'Data Selection' tab
			- While this is checked, left click on any not currently used data point (non-highlighted points in the plots) to add that period to the inversion.
				- The selected point should become highlighted for all sites
			- Left click to remove points
		- You can remove stations from an inversion by selecting the station from the list and clicking the right arrow button.
		- Stations can be added back in by doing the opposite.
			- Note that if you change the stations either by adding/removing or changing the order (e.g., with the 'Sort Sites' menu), it is recommended that you write out a new list file (found in the 'Write' menu at the top) to ensure that your list files and inversion data files are consistent	
			- If you load a data_plot instance and the highlighted circles are plotting off of the unhighlighted ones, then your files are inconsistent (either the site order is wrong, or the data is different (e.g., rotations are different))
		- Errors to be used to input into an inversion can be manipulated in 3 ways:
			- Hard reset
				- Go to the 'Error Manipulations' Tab.
				- Enter the percent error floor you would like to apply to each data component (in decimal, e.g., 5% error is 0.05)
				- Hit the 'Reset Errors' button
				- All errors are now set to 5% of the data value
					- For ZXX and ZYY data, the calculation is based on the ZXY and ZYX components to avoid low or zero error values
			- Outlier detection
				- The 'Regulate Errors' button will reset the errors to the error floor and then attempt to flag outliers
				- A smooth fit curve to the data is calculated, and data points that lie off this curve by a certain amount will be flagged
				- The first float number below the 'Regulate Errors' defines how smooth the fitted curve is. Lower numbers will allow for faster changes, which will mean fewer points are flagged as outliers.
					- Default value is 1, anywhere between 0.8 and 1 is usually good.
				- Second float number below the 'Regulate Errors' Button controls how high the errors are set to. It multiplies the difference between the actual data point and the smooth fitted curve.
					- Default is 2, but values between 1 and 1.5 are generally good.
			- Manual error mapping
				- Go to the 'Error Multipliers' tab
				- Here you will see the multipliers that are being used to calculate the errors (used error = raw error * multiplier)
				- If you used the 'Reset Errors' button, these should all be equal to 1
				- From here, you can go site by site, period by period, and component by component to manually tweak the inversion errors
				- Double clicking on a value and editing it will change that value
				- Holding SHIFT and editing a value will change the values of that station and component for all periods (i.e., going down the table)
				- Holding CTRL and editing a value will change the values for that site and period, for all components (i.e., goin across the table)
				- Holding ALT and editing a value will change the values for that component and period for all sites
				- These buttons can be used together, e.g., holding SHIFT and CTRL will edit the errormaps for all periods and components for a single site, while holding SHIFT and ALT will edit all periods and all sites for a single component
		- Writing out an inversion data file
			- Click the 'Inversion Type' menu at the top
			- Select the type of inversion you would like to run
				- Options 1-5 are for 3-D, while 8-10 are for 2-D
			- Click the 'Write' menu button
			- Click 'Data File'
			- Select the data format you would like to write to
				- Note some inversion types are only valid for some file formats
					- 2-D and phase tensor inversions are only valid in ModEM

	- model_viewer
		- Loads models and data files to view in 2-D or 3-D
		- Run via:
			model_viewer data_file model_file
			(order of files doesn't matter)
		- Note that this is made for 3-D file formats, and so while it will read in a 2-D model, it will extrapolate it out in the strike direction.
			- On the same note, the station locations as specified in the 2-D ModEM data format are not the same as those for 3-D, and so loading in 2-D ModEM data and model files may result in the locations plotting off of the model.