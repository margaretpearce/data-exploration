import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from controllers.data_driver import DataDriver
from configuration import const_types

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

    def test_get_features_list_label(self):
        actual = self.driver.get_features_list()
        assert self.driver.label_column in actual

    def test_get_features_list_id(self):
        actual = self.driver.get_features_list()
        assert self.driver.id_column in actual

    def test_get_data_type_boolean(self):
        actual = self.driver.get_data_type("Survived")
        expected = const_types.DATATYPE_BOOLEAN
        assert actual == expected

    def test_get_data_type_int(self):
        actual = self.driver.get_data_type("SibSp")
        expected = const_types.DATATYPE_INTEGER
        assert actual == expected

    def test_get_data_type_float(self):
        actual = self.driver.get_data_type("Age")
        expected = const_types.DATATYPE_FLOAT
        assert actual == expected

    def test_get_data_type_string(self):
        actual = self.driver.get_data_type("Name")
        expected = const_types.DATATYPE_STRING
        assert actual == expected

    def test_get_variable_type_continous(self):
        actual = self.driver.get_variable_type("Age")
        expected = const_types.VARTYPE_CONTINUOUS
        assert actual == expected



