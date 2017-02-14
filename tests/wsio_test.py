from pyMT import IO
from pyMT.data_structures import RawData
from pyMT.data_structures import Data
from pyMT.data_structures import Dataset
import pyMT.utils as utils
from pyMT.WSExceptions import WSFileError
from nose.tools import assert_equal
from nose.tools import assert_raises
import numpy as np

# from nose.tools import ok_


testlist = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion' \
           r'\Regions\abi-gren\Old\abi0\rmsites.lst'
datafile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion' \
           r'\Regions\abi-gren\Old\abi0\rmsitesNew_1.data'
datpath = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\abi-gren\Old\j2'

outfile = r'D:\pythonProgs\Inversion\testfile.data'
datafile_mistmatch = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion' \
                     r'\Regions\abi-gren\Old\abi0_7\abi0_7.data'


# Can create a subclass from this later and just modify the setup to also add or subtract something
# from the dataset to make sure that it is properly reflected in the written data.
class TestStraightWriteDataset(object):
    """
    Test to make sure that when the data read into a Dataset is written out and read
    back in again, that the results are identical.
    """
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(listfile=testlist, datafile=datafile, datpath=datpath)
        cls.dataset.data.write(outfile)
        cls.dataset2 = Dataset(listfile=testlist, datafile=outfile, datpath=datpath)

    def test_comps_are_equal(self):
        assert_equal(self.dataset.data.components, self.dataset2.data.components)

    def test_pers_equal(self):
        assert np.all(self.dataset.data.periods == self.dataset2.data.periods)

    def test_site_names_equal(self):
        assert_equal(self.dataset.data.site_names, self.dataset2.data.site_names)

    def test_locations_equal(self):
        assert np.all(self.dataset.data.locations == self.dataset2.data.locations)

    def test_data_equal(self):
        for name in self.dataset.data.site_names:
            site1 = self.dataset.data.sites[name]
            site2 = self.dataset2.data.sites[name]
            for comp in self.dataset.data.components:
                np.testing.assert_allclose(utils.truncate(site1.data[comp]),
                                           utils.truncate(site2.data[comp]), 10e-5)

    def test_errors_equal(self):
        for name in self.dataset.data.site_names:
            site1 = self.dataset.data.sites[name]
            site2 = self.dataset2.data.sites[name]
            for comp in self.dataset.data.components:
                np.testing.assert_allclose(utils.truncate(site1.errors[comp]),
                                           utils.truncate(site2.errors[comp]), 10e-5)

    def test_errmap_equal(self):
        for name in self.dataset.data.site_names:
            site1 = self.dataset.data.sites[name]
            site2 = self.dataset2.data.sites[name]
            for comp in self.dataset.data.components:
                np.testing.assert_allclose(utils.truncate(site1.errmap[comp]),
                                           utils.truncate(site2.errmap[comp]), 10e-5)


class TestStraightWriteData(object):
    """
    Test to make sure that when a Data instance is created from a data file
    and Immediately written out, the the results are identical.
    """
    @classmethod
    def setUpClass(cls):
        cls.data = Data(datafile=datafile, listfile=testlist)
        cls.data.write(outfile)
        cls.data2 = Data(datafile=outfile, listfile=testlist)

    def test_comps_are_equal(self):
        assert_equal(self.data.components, self.data2.components)

    def test_pers_equal(self):
        assert np.all(self.data.periods == self.data2.periods)

    def test_site_names_equal(self):
        assert_equal(self.data.site_names, self.data2.site_names)

    def test_locations_equal(self):
        np.testing.assert_allclose(self.data.locations, self.data2.locations)

    def test_data_equal(self):
        for name in self.data.site_names:
            site1 = self.data.sites[name]
            site2 = self.data2.sites[name]
            for comp in self.data.components:
                np.testing.assert_allclose(utils.truncate(site1.data[comp]),
                                           utils.truncate(site2.data[comp]), 10e-5)

    def test_errors_equal(self):
        for name in self.data.site_names:
            site1 = self.data.sites[name]
            site2 = self.data2.sites[name]
            for comp in self.data.components:
                try:
                    np.testing.assert_allclose(utils.truncate(site1.errors[comp]),
                                               utils.truncate(site2.errors[comp]), 10e-5)
                except AssertionError as e:
                    print(site1.errors[comp])
                    print(site2.errors[comp])
                    raise(e)

    def test_errmap_equal(self):
        for name in self.data.site_names:
            site1 = self.data.sites[name]
            site2 = self.data2.sites[name]
            for comp in self.data.components:
                np.testing.assert_allclose(utils.truncate(site1.errmap[comp]),
                                           utils.truncate(site2.errmap[comp]), 10e-5)


class TestWSIOWithGoodFile_RawData(object):
    """Summary
    Test that a RawData object initialized with a real file is correct.
    """
    @classmethod
    def setUpClass(cls):
        cls.data = RawData(testlist, datpath)

    def test_read_dats_site_names(self):
        assert_equal(set([site.name for site in self.data.sites.values()]),
                     set(self.data.site_names))

    def test_read_dats_periods(self):
        for site in self.data.sites.values():
            try:
                assert set(site.periods).issubset(set(self.data.master_periods.keys()))
            except AssertionError as e:
                print(site.name)
                raise(e)

    def test_azi_set_to_zero(self):
        for site in self.data.sites.values():
            try:
                assert_equal(site.azimuth, 0)
            except AssertionError as e:
                print(site.name)
                raise(e)

    def test_data_length(self):
        for site in self.data.sites.values():
            for comp in site.components:
                assert_equal(len(site.periods), len(site.data[comp]))


class TestWSIOWithGoodFile_Data(object):
    """Test that a Data object is initilized correctly when using proper files.
    """
    @classmethod
    def setUpClass(cls):
        cls.data = Data(datafile=datafile, listfile=testlist)

    def test_data_periods_are_identical(self):
        for site in self.data.sites.values():
            assert all(site.periods == self.data.periods)

    def test_data_names_are_identical(self):
        for site in self.data.sites.values():
            assert site.name in self.data.site_names

    def test_data_components_are_identical(self):
        for site in self.data.sites.values():
            assert_equal(site.components, self.data.components)

    def test_data_all_sites_loosely_equal(self):
        for site1 in self.data.sites.values():
            for site2 in self.data.sites.values():
                assert site1.loosely_equal(site2) and site2.loosely_equal(site1)


class TestIOWithBadFiles(object):
    """Test that proper exeptions are raised when attempting to initialize data_structures with
    bad or nonexistant files.
    """

    def test_readDirect_sites_nonexistant_list(self):
        with assert_raises(WSFileError):
            IO.read_sites('this_is_not_a_file.lst')

    def test_readDirect_data_nonexistant_data(self):
        with assert_raises(WSFileError):
            IO.read_data(datafile='this_is_not_a_file.lst')

    def test_readDirect_raw_data_nonexistant_data(self):
        sites = IO.read_sites('badlist.lst')
        with assert_raises(WSFileError):
            IO.read_raw_data(site_names=sites)

    def test_readFromObj_raw_data_nonexistant_data(self):
        with assert_raises(WSFileError):
            RawData(listfile='badlist.lst')

    def test_readFromObj_data_nonexistant_data(self):
        with assert_raises(WSFileError):
            Data(datafile='this_is_not_a_file.lst')

    def test_read_mismatched_files(self):
        with assert_raises(WSFileError):
            Data(datafile=datafile_mistmatch, listfile=testlist)
