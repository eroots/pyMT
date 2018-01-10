from pyMT.data_structures import RawData
from pyMT.data_structures import Data
from pyMT.data_structures import Site
from nose.tools import assert_equal


class TestEmptySite(object):
    @classmethod
    def setUpClass(cls):
        cls.site = Site()

    def test_empty_site_name(self):
        assert_equal(self.site.name, '')

    def test_empty_site_data(self):
        assert_equal(self.site.data, {})

    def test_empty_site_periods(self):
        assert_equal(self.site.periods, None)

    def test_empty_site_locations(self):
        assert_equal(self.site.locations, {})

    def test_empty_site_errors(self):
        assert_equal(self.site.errors, {})

    def test_empty_site_components(self):
        assert_equal(self.site.components, [])

    def test_empty_site_errmap(self):
        assert_equal(self.site.errmap, {})

    def test_empty_site_used_errors(self):
        assert_equal(self.site.used_error, {})


class TestEmptyData(object):
    @classmethod
    def setUpClass(cls):
        cls.data = Data()

    def test_init_empty_site_names(self):
        assert_equal(self.data.site_names, [])

    def test_init_empty_site_data(self):
        assert_equal(self.data.sites, {})

    def test_init_empty_datafile(self):
        assert_equal(self.data.datafile, '')

    def test_init_empty_periods(self):
        assert_equal(self.data.periods, [])

    def test_init_empty_locations(self):
        assert_equal(self.data.locations, [])

    def test_init_empty_components(self):
        assert_equal(self.data.components, [])


class TestEmptyRawData(object):
    @classmethod
    def setUpClass(cls):
        cls.data = RawData()

    def test_init_empty_site_names(self):
        assert_equal(self.data.site_names, [])

    def test_init_empty_site_data(self):
        assert_equal(self.data.sites, {})

    def test_init_empty_datfile(self):
        assert_equal(self.data.datpath, '')

    def test_init_empty_periods(self):
        assert_equal(self.data.master_periods, {})
        assert_equal(self.data.narrow_periods, {})

    def test_init_empty_locations(self):
        assert_equal(self.data.locations, [])

