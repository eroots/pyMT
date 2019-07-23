data_structures
===============

.. _Dataset:

Dataset
-------

.. _Data:

Data
----

.. _Error Floors:

Error Floors
------------

Error floors may be set in a Data object by setting the corresponding value in the error_floors attribute.
Data.error_floors initialized as::

	Data.error_floors = {'Off-Diagonal Impedance': 0.05,
                         'Diagonal Impedance': 0.075,
                         'Tipper': 0.05,
                         'Rho': 0.05,
                         'Phase': 0.03}

The corresponding entries may be modified as any dictionary and then accesing the apply_error_floor method. For instance, to change the error floor of the ZXY and ZYX components to 7.5%, use::
	
	data.error_floors['Off-Diagonal Impedance'] = 0.075
	data.apply_error_floor()

.. _RawData:

RawData
-------

.. _Response:

Response
--------

.. _Site:

Site
----

.. _Model:

Model
-----