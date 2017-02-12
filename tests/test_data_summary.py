# Ugly hack to fix imports
import sys, os
sys.path.insert(0, os.path.abspath('..'))

from controllers.data_summary import DataSummary
from model.summary import Summary


def test_load_summary_json():
    dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
    summary = DataSummary(dataset)

    # Confirm the data set can be loaded into a Summary object
    summary_loaded = summary.load_summary_json()
    actual = type(summary_loaded)
    expected = Summary

    assert actual == expected

def test_get_num_records():
    dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
    summary = DataSummary(dataset)
    summary.load_data()

    expected = 891
    actual = summary.get_num_records()

    assert expected == actual

def test_get_num_features():
    dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
    summary = DataSummary(dataset)
    summary.load_data()

    expected = 12
    actual = summary.get_num_features()

    assert expected == actual

def test_count_missing():
    dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
    summary = DataSummary(dataset)
    summary.load_data()

    num_records = 891
    missing = summary.count_missing(num_records)
    max_missing = len(missing)
    missing_sum = 0

    for i in range(max_missing):
        missing_sum += missing.get(i)

    assert missing_sum == num_records

def test_get_sample_returns_list():
    dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
    summary = DataSummary(dataset)
    summary.load_data()

    sample = summary.get_sample(5, summary.get_features_list())
    expected = list
    actual = type(sample)

    assert expected == actual

def test_get_sample_size():
    dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
    summary = DataSummary(dataset)
    summary.load_data()

    sample_size = 5
    sample = summary.get_sample(sample_size, summary.get_features_list())
    expected = sample_size
    actual = len(sample)

    assert expected == actual