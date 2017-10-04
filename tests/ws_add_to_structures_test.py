from pyMT import IO
import numpy as np
from pyMT.data_structures import Dataset
from pyMT.data_structures import Data
from pyMT.WSExceptions import WSFileError
from nose.tools import assert_equal
from nose.tools import assert_raises
import copy
import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
testlist = os.path.join(THIS_DIR, os.pardir, 'test_data', 'rmsites.lst')
datafile = os.path.join(THIS_DIR, os.pardir, 'test_data', 'rmsitesNew_1.data')
datpath = os.path.join(THIS_DIR, os.pardir, 'test_data', 'j2')


class TestWSSiteAddMethods(object):
    """Summary
    Test that a RawData object initialized with a real file is correct.
    """
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(listfile=testlist, datafile=None, responsefile=None, datpath=datpath)

    def setUp(self):
        self.data = copy.deepcopy(self.dataset.data)

    def test_adding_periods_to_site(self):
        site_name = self.dataset.raw_data.site_names[0]
        periods = self.dataset.raw_data.sites[site_name].periods[0:6]
        old_site = self.dataset.data.sites[site_name]
        raw_site = self.dataset.raw_data.sites[site_name]
        to_add = self.dataset.raw_data.get_data(sites=site_name, periods=periods,
                                                components=old_site.components)
        old_site.add_periods(to_add[site_name])
        for ii, period in enumerate(old_site.periods):
            if period not in list(old_site.periods):
                ind = np.argmin(old_site.periods - period)
                for comp in old_site.components:
                    assert (old_site.data[comp][ii] == raw_site.data[comp][ind])
