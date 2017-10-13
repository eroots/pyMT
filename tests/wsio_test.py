from pyMT import IO
from pyMT.data_structures import RawData
from pyMT.data_structures import Data
from pyMT.data_structures import Dataset
import pyMT.utils as utils
from pyMT.WSExceptions import WSFileError
from nose.tools import assert_equal
from nose.tools import assert_raises
import numpy as np
import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WS_test_list = os.path.join(THIS_DIR, os.pardir, 'test_data', 'rmsites.lst')
WS_data_file = os.path.join(THIS_DIR, os.pardir, 'test_data', 'rmsitesNew_1.data')
ModEM_test_list = os.path.join(THIS_DIR, os.pardir, 'test_data', 'ModEM_test_list.lst')
ModEM_test_data = os.path.join(THIS_DIR, os.pardir, 'test_data', 'ModEM_test_data.dat')
MARE2DEM_test_list = os.path.join(THIS_DIR, os.pardir, 'test_data', 'MARE2DEM_test_list.lst')
MARE2DEM_test_data = os.path.join(THIS_DIR, os.pardir, 'test_data', 'MARE2DEM_test_data.emdata')
datpath = os.path.join(THIS_DIR, os.pardir, 'test_data', 'j2')
badlist = os.path.join(THIS_DIR, os.pardir, 'test_data', 'badlist.lst')
outfile_WS = os.path.join(THIS_DIR, os.pardir, 'test_data', 'testfile_WS.data')
outfile_ModEM = os.path.join(THIS_DIR, os.pardir, 'test_data', 'testfile_ModEM.dat')
outfile_MARE2DEM = os.path.join(THIS_DIR, os.pardir, 'test_data', 'testfile_MARE2DEM.emdata')
datafile_mistmatch = os.path.join(THIS_DIR, os.pardir, 'test_data', 'allsites.data')


# Can create a subclass from this later and just modify the setup to also add or subtract something
# from the dataset to make sure that it is properly reflected in the written data.
class TestStraightWriteDataset_WS(object):
    """
    Test to make sure that when the data read into a Dataset is written out and read
    back in again, that the results are identical.
    """
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(listfile=WS_test_list, datafile=WS_data_file, datpath=datpath)
        cls.dataset.data.write(outfile_WS)
        cls.dataset2 = Dataset(listfile=WS_test_list, datafile=WS_data_file, datpath=datpath)

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


class TestStraightWriteData_WS(object):
    """
    Test to make sure that when a Data instance is created from a data file
    and Immediately written out, the the results are identical.
    """
    @classmethod
    def setUpClass(cls):
        cls.data = Data(datafile=WS_data_file, listfile=WS_test_list)
        cls.data.write(outfile_WS)
        cls.data2 = Data(datafile=outfile_WS, listfile=WS_test_list)

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
        cls.data = RawData(WS_test_list, datpath)

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


class TestIOWithGoodFile_WSData(object):
    """Test that a Data object is initilized correctly when using proper files.
    """
    @classmethod
    def setUpClass(cls):
        cls.data = Data(datafile=WS_data_file, listfile=WS_test_list)

    def test_data_periods_are_identical(self):
        for site in self.data.sites.values():
            assert all(site.periods == self.data.periods)

    def test_data_names_are_identical(self):
        for site in self.data.sites.values():
            assert site.name in self.data.site_names

    def test_data_components_are_identical(self):
        for site in self.data.sites.values():
            assert_equal(set(site.components), set((self.data.components)))

    def test_data_all_sites_loosely_equal(self):
        for site1 in self.data.sites.values():
            for site2 in self.data.sites.values():
                assert site1.loosely_equal(site2) and site2.loosely_equal(site1)


class TestIOWithGoodFile_ModEMData(TestIOWithGoodFile_WSData):
    @classmethod
    def setUpClass(cls):
        cls.data = Data(datafile=ModEM_test_data, listfile=ModEM_test_list)


class TestWriteDataset_ModEM(TestStraightWriteDataset_WS):
    @classmethod
    def setUpClass(cls):
        cls.data = Data(datafile=ModEM_test_data, listfile=ModEM_test_list)
        cls.data.write(outfile_ModEM)
        cls.data2 = Data(datafile=outfile_ModEM, listfile=ModEM_test_list)


class TestWriteData_ModEM(TestStraightWriteData_WS):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(listfile=ModEM_test_list, datafile=ModEM_test_data, datpath=datpath)
        cls.dataset.data.write(outfile_ModEM)
        cls.dataset2 = Dataset(listfile=ModEM_test_list, datafile=ModEM_test_data, datpath=datpath)


class TestIOWithGoodFile_MARE2DEMData(TestIOWithGoodFile_WSData):
    @classmethod
    def setUpClass(cls):
        cls.data = Data(datafile=MARE2DEM_test_data, listfile=MARE2DEM_test_list)


class TestWriteDataset_MARE2DEM(TestStraightWriteDataset_WS):
    @classmethod
    def setUpClass(cls):
        cls.data = Data(datafile=MARE2DEM_test_data, listfile=MARE2DEM_test_list)
        cls.data.write(outfile_MARE2DEM)
        cls.data2 = Data(datafile=outfile_MARE2DEM, listfile=MARE2DEM_test_list)


class TestWriteData_MARE2DEM(TestStraightWriteData_WS):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(listfile=MARE2DEM_test_list, datafile=MARE2DEM_test_data, datpath=datpath)
        cls.dataset.data.write(outfile_MARE2DEM)
        cls.dataset2 = Dataset(listfile=MARE2DEM_test_list, datafile=MARE2DEM_test_data, datpath=datpath)


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
        sites = IO.read_sites(badlist)
        with assert_raises(WSFileError):
            IO.read_raw_data(site_names=sites)

    def test_readFromObj_raw_data_nonexistant_data(self):
        with assert_raises(WSFileError):
            RawData(listfile=badlist)

    def test_readFromObj_data_nonexistant_data(self):
        with assert_raises(WSFileError):
            Data(datafile='this_is_not_a_file.lst')

    def test_read_mismatched_files(self):
        with assert_raises(WSFileError):
            Data(datafile=datafile_mistmatch, listfile=WS_test_list)
