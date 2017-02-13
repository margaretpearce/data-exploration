# Ugly hack to fix imports
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from controllers.data_summary import DataSummary
from model.summary import Summary


class TestDataSummary:
    @classmethod
    def setup_class(cls):
        cls.dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
        cls.summary = DataSummary(cls.dataset)
        cls.summary.load_data()

    def test_load_summary_json(self):
        # Confirm the data set can be loaded into a Summary object
        summary_loaded = self.summary.load_summary_json()
        actual = type(summary_loaded)
        expected = Summary
        assert actual == expected

    def test_get_num_records(self):
        expected = 891
        actual = self.summary.get_num_records()
        assert expected == actual

    def test_get_num_features(self):
        expected = 12
        actual = self.summary.get_num_features()
        assert expected == actual

    def test_count_missing(self):
        num_records = 891
        missing = self.summary.count_missing(num_records)
        max_missing = len(missing)
        missing_sum = 0

        for i in range(max_missing):
            missing_sum += missing.get(i)

        assert missing_sum == num_records

    def test_get_sample_returns_list(self):
        sample = self.summary.get_sample(5, self.summary.get_features_list())
        expected = list
        actual = type(sample)
        assert expected == actual

    def test_get_sample_size(self):
        sample_size = 5
        sample = self.summary.get_sample(sample_size, self.summary.get_features_list())
        expected = sample_size
        actual = len(sample)
        assert expected == actual
