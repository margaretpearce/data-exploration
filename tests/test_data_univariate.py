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

    def test_get_median_float(self):
        expected = 28
        actual = self.univariate.get_median("Age")
        assert expected == actual

    def test_get_median_binary(self):
        expected = 0
        actual = self.univariate.get_median("Survived")
        assert expected == actual

    def test_get_median_string(self):
        expected = None
        actual = self.univariate.get_median("Name")
        assert expected == actual

    def test_get_mode_int(self):
        expected = "0"
        actual = self.univariate.get_mode("SibSp").strip()
        assert expected == actual

    def test_get_mode_float(self):
        expected = "24.0"
        actual = self.univariate.get_mode("Age").strip()
        assert expected == actual

    def test_get_mode_binary(self):
        expected = "0"
        actual = self.univariate.get_mode("Survived").strip()
        assert expected == actual

    def test_get_mode_string(self):
        expected = "S"
        actual = self.univariate.get_mode("Embarked").strip()
        assert expected == actual

    def test_get_max_int(self):
        expected = 8
        actual = self.univariate.get_max("SibSp")
        assert expected == actual

    def test_get_max_float(self):
        expected = 80
        actual = self.univariate.get_max("Age")
        assert expected == actual

    def test_get_max_binary(self):
        expected = 1
        actual = self.univariate.get_max("Survived")
        assert expected == actual

    def test_get_max_string(self):
        expected = None
        actual = self.univariate.get_max("Name")
        assert expected == actual

    def test_get_min_int(self):
        expected = 0
        actual = self.univariate.get_min("SibSp")
        assert expected == actual

    def test_get_min_float(self):
        expected = 0.42
        actual = self.univariate.get_min("Age")
        assert expected == actual

    def test_get_min_binary(self):
        expected = 0
        actual = self.univariate.get_min("Survived")
        assert expected == actual

    def test_get_min_string(self):
        expected = None
        actual = self.univariate.get_min("Name")
        assert expected == actual

    def test_get_stddev_int(self):
        expected = 1.103
        actual = round(self.univariate.get_stddev("SibSp"), 3)
        assert expected == actual

    def test_get_stddev_float(self):
        expected = 14.526
        actual = round(self.univariate.get_stddev("Age"), 3)
        assert expected == actual

    def test_get_stddev_binary(self):
        expected = 0.487
        actual = round(self.univariate.get_stddev("Survived"), 3)
        assert expected == actual

    def test_get_stddev_string(self):
        expected = None
        actual = self.univariate.get_stddev("Name")
        assert expected == actual

    def test_get_variance_int(self):
        expected = 1.216
        actual = round(self.univariate.get_variance("SibSp"), 3)
        assert expected == actual

    def test_get_variance_float(self):
        expected = 211.019
        actual = round(self.univariate.get_variance("Age"), 3)
        assert expected == actual

    def test_get_variance_binary(self):
        expected = 0.237
        actual = round(self.univariate.get_variance("Survived"), 3)
        assert expected == actual

    def test_get_variance_name(self):
        expected = None
        actual = self.univariate.get_variance("Name")
        assert expected == actual

    def test_get_quantile25_int(self):
        expected = 0.000
        actual = round(self.univariate.get_quantile25("SibSp"), 3)
        assert expected == actual

    def test_get_quantile25_float(self):
        expected = 20.125
        actual = round(self.univariate.get_quantile25("Age"), 3)
        assert expected == actual

    def test_get_quantile25_binary(self):
        expected = 0.000
        actual = round(self.univariate.get_quantile25("Survived"), 3)
        assert expected == actual

    def test_get_quantile25_string(self):
        expected = None
        actual = self.univariate.get_quantile25("Name")
        assert expected == actual

    def test_get_quantile75_int(self):
        expected = 1.000
        actual = round(self.univariate.get_quantile75("SibSp"), 3)
        assert expected == actual

    def test_get_quantile75_float(self):
        expected = 38.000
        actual = round(self.univariate.get_quantile75("Age"), 3)
        assert expected == actual

    def test_get_quantile75_binary(self):
        expected = 1.000
        actual = round(self.univariate.get_quantile75("Survived"), 3)
        assert expected == actual

    def test_get_quantile75_string(self):
        expected = None
        actual = self.univariate.get_quantile75("Name")
        assert expected == actual

    def test_get_iqr_int(self):
        field_name = "SibSp"
        expected = round(self.univariate.get_quantile75(field_name) - self.univariate.get_quantile25(field_name), 3)
        actual = round(self.univariate.get_iqr(field_name), 3)
        assert expected == actual

    def test_get_iqr_float(self):
        field_name = "Age"
        expected = round(self.univariate.get_quantile75(field_name) - self.univariate.get_quantile25(field_name), 3)
        actual = round(self.univariate.get_iqr(field_name), 3)
        assert expected == actual

    def test_get_iqr_binary(self):
        field_name = "Survived"
        expected = round(self.univariate.get_quantile75(field_name) - self.univariate.get_quantile25(field_name), 3)
        actual = round(self.univariate.get_iqr(field_name), 3)
        assert expected == actual

    def test_get_iqr_string(self):
        field_name = "Name"
        expected = None
        actual = self.univariate.get_iqr(field_name)
        assert expected == actual

    def test_get_skew_int(self):
        expected = 3.695
        actual = round(self.univariate.get_skew("SibSp"), 3)
        assert expected == actual

    def test_get_skew_float(self):
        expected = 0.389
        actual = round(self.univariate.get_skew("Age"), 3)
        assert expected == actual

    def test_get_skew_binary(self):
        expected = 0.479
        actual = round(self.univariate.get_skew("Survived"), 3)
        assert expected == actual

    def test_get_skew_string(self):
        expected = None
        actual = self.univariate.get_skew("Name")
        assert expected == actual