.. _Recipes:

Recipes
=======

All the following recipes assume pyMT modules have been imported as follows::
	
	import pyMT.data_structures as ds
	import pyMT.utils as utils

Adding Data to an Existing Data File
------------------------------------

Adding new periods to an existing data file is supported through the :ref:`Data Plot` GUI. However, it is assumed that the list file and data file used are consistent, i.e., that they have the same stations and in the same order.
Therefore, adding new stations to an existing data file must be done using the pyMT API. 

This example uses the :ref:`Data`, and :ref:`RawData` classes as well as the Data.get_data and Data.add_site methods.

Assuming the data file we want to modify is 'demo.dat', and the list file we want to pull in new sites from is 'raw/all.lst'::
	
	data = ds.Data('demo.dat')  # Load in the data file
	raw_data = ds.RawData('raw/all.lst') # Load in the raw data
	# Get the new site to add with the same periods and components as 'data'
	to_add = raw_data.get_data(periods=data.periods, 
							   components=data.components,
							   sites='test_1')
    data.add_site(to_add['test_1']) # Add the new site to the data object
    data.write('new_data', file_format='modem')  # Write the new data out.