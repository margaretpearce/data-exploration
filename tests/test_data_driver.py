import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from controllers.data_driver import DataDriver
from configuration import const_types
from configuration import paths

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

    def test_get_variable_type_categorical(self):
        actual = self.driver.get_variable_type("Embarked")
        expected = const_types.VARTYPE_CATEGORICAL
        assert actual == expected

    def test_get_variable_type_binary(self):
        actual = self.driver.get_variable_type("Survived")
        expected = const_types.VARTYPE_BINARY
        assert actual == expected

    def test_get_percent_unique_all(self):
        actual = self.driver.get_percent_unique("Name")
        expected = 1
        assert actual == expected

    def test_get_percent_unique_some(self):
        actual = self.driver.get_percent_unique("Embarked")
        expected = self.driver.get_count_unique("Embarked") / float(self.driver.data["Embarked"].count())
        assert actual == expected

    def test_get_percent_unique_float(self):
        actual = self.driver.get_percent_unique("Age")
        expected = self.driver.get_count_unique("Age") / float(self.driver.data["Age"].count())
        assert actual == expected

    def test_get_percent_unique_missingvals(self):
        actual = self.driver.get_percent_unique("Cabin")
        expected = self.driver.get_count_unique("Cabin") / float(self.driver.data["Cabin"].count())
        assert actual == expected

    def test_get_count_unique_categorical(self):
        actual = self.driver.get_count_unique("Sex")
        expected = 2
        assert actual == expected

    def test_get_count_unique_binary(self):
        actual = self.driver.get_count_unique("Survived")
        expected = 2
        assert actual == expected

    def test_get_count_unique_missingvals(self):
        actual = self.driver.get_count_unique("Embarked")
        expected = 4 # Three real values, one "null"
        assert actual == expected

    def test_check_uniques_for_graphing_valid(self):
        actual = self.driver.check_uniques_for_graphing("Embarked")
        expected = True
        assert actual == expected

    def test_check_uniques_for_graphing_too_high_count(self):
        actual = self.driver.check_uniques_for_graphing("Ticket")
        expected = False
        assert actual == expected

    def test_check_uniques_for_graphing_too_high_perc(self):
        actual = self.driver.check_uniques_for_graphing("Name")
        expected = False
        assert actual == expected

    def test_get_error_msg_none(self):
        actual = self.driver.get_error_msg()
        expected = None
        assert actual == expected

    def test_load_json_summary(self):
        actual = self.driver.load_json(paths.SUMMARY_SUFFIX)
        assert actual is not None

    def test_load_json_features(self):
        actual = self.driver.load_json(paths.FEATURES_SUFFIX)
        assert actual is not None

    def test_load_json_interactions(self):
        actual = self.driver.load_json(paths.INTERACTIONS_SUFFIX)
        assert actual is not None

    def test_load_json_nonexistent(self):
        actual = self.driver.load_json("other.json")
        assert actual is None