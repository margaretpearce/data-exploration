# Ugly hack to fix imports
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from controllers.data_univariate import DataUnivariate
from model.features import Features


class TestDataUnivariate:
    @classmethod
    def setup_class(cls):
        cls.dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
        cls.univariate = DataUnivariate(cls.dataset)
        cls.univariate.load_data()

    def test_load_univariate_json(self):
        features_loaded = self.univariate.load_features_json()
        actual = type(features_loaded)
        expected = Features
        assert actual == expected

    def test_count_binary(self):
        expected = 891
        actual = self.univariate.get_count("Survived")
        assert actual == expected

    def test_count_string(self):
        expected = 891
        actual = self.univariate.get_count("Name")
        assert actual == expected

    def test_count_int(self):
        expected = 891
        actual = self.univariate.get_count("SibSp")
        assert actual == expected

    def test_count_float(self):
        expected = 891
        actual = self.univariate.get_count("Fare")
        assert actual == expected

    def test_count_with_missing_vals(self):
        expected = 714
        actual = self.univariate.get_count("Age")
        assert actual == expected

    def test_get_count_missing_string(self):
        expected = 687
        actual = self.univariate.get_count_missing("Cabin")
        assert actual == expected

    def test_get_count_missing_float(self):
        expected = 177
        actual = self.univariate.get_count_missing("Age")
        assert actual == expected

    def test_count_plus_missing_equals_length(self):
        not_missing_count = self.univariate.get_count("Age")
        missing_count = self.univariate.get_count_missing("Age")
        total_count_expected = 891
        assert total_count_expected == missing_count + not_missing_count

    def test_count_missing_none_missing(self):
        expected = 0
        actual = self.univariate.get_count_missing("PassengerId")
        assert actual == expected

