import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from controllers.data_driver import DataDriver


class TestDataDriver:
    @classmethod
    def setup_class(cls):
        cls.dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
        cls.driver = DataDriver(cls.dataset)
        cls.driver.load_data()

    def test_load_data(self):
        actual = self.driver.load_data()
        expected = True
        assert actual == expected

    def test_format_rounded_string_none(self):
        actual = self.driver.format_rounded_string(None)
        expected = None
        assert actual == expected

    def test_format_rounded_string_float(self):
        actual = self.driver.format_rounded_string(3.1415826)
        expected = "3.142"
        assert actual == expected

    def test_format_rounded_string_int(self):
        actual = self.driver.format_rounded_string(3)
        expected = "3"
        assert actual == expected