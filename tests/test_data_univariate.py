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

    def test_count_percent_missing(self):
        num_missing = self.univariate.get_count_missing("Age")
        num_rows = 891
        expected = 100 * num_missing / float(num_rows)
        actual = self.univariate.get_percent_missing("Age")
        assert actual == expected

    def test_feat_is_numeric_string(self):
        expected = False
        actual = self.univariate.feat_is_numeric("Name")
        assert actual == expected

    def test_feat_is_numeric_int(self):
        expected = True
        actual = self.univariate.feat_is_numeric("SibSp")
        assert actual == expected

    def test_feat_is_numeric_float(self):
        expected = True
        actual = self.univariate.feat_is_numeric("Age")
        assert actual == expected

    def test_feat_is_numeric_binary(self):
        expected = True
        actual = self.univariate.feat_is_numeric("Survived")
        assert actual == expected

    def test_get_average_binary(self):
        expected = 0.384
        actual = round(self.univariate.get_average("Survived"), 3)
        assert expected == actual

    def test_get_average_float(self):
        expected = 29.699
        actual = round(self.univariate.get_average("Age"), 3)
        assert expected == actual

    def test_get_average_string(self):
        expected = None
        actual = self.univariate.get_average("Name")
        assert expected == actual

    def test_get_average_int(self):
        expected = 0.523
        actual = round(self.univariate.get_average("SibSp"), 3)
        assert expected == actual

    def test_get_median_int(self):
        expected = 0
        actual = self.univariate.get_median("SibSp")
        assert expected == actual