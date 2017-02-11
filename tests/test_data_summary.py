from controllers.data_summary import DataSummary
from model.summary import Summary


def load_summary_json_test():
    dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
    summary = DataSummary(dataset)

    # Confirm the data set can be loaded into a Summary object
    summary_loaded = summary.load_summary_json()
    actual = type(summary_loaded)
    expected = Summary

    assert actual == expected

def get_num_records_test():
    dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
    summary = DataSummary(dataset)
    summary.load_data()

    expected = 891
    actual = summary.get_num_records()

    assert expected == actual

def get_num_features_test():
    dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
    summary = DataSummary(dataset)
    summary.load_data()

    expected = 12
    actual = summary.get_num_features()

    assert expected == actual

def count_missing_test():
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

def get_sample_returns_list_test():
    dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
    summary = DataSummary(dataset)
    summary.load_data()

    sample = summary.get_sample(5, summary.get_features_list())
    expected = list
    actual = type(sample)

    assert expected == actual

def get_sample_size_test():
    dataset = ['titanic.csv', 'Titanic', 'PassengerId', 'Survived', False]
    summary = DataSummary(dataset)
    summary.load_data()

    sample_size = 5
    sample = summary.get_sample(sample_size, summary.get_features_list())
    expected = sample_size
    actual = len(sample)

    assert expected == actual

if __name__ == '__main__':
    load_summary_json_test()
    get_num_records_test()
    get_num_features_test()
    count_missing_test()
    get_sample_returns_list_test()
    get_sample_size_test()
    print("done")