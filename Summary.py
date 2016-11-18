class Summary(object):
    def __init__(self, name, num_records, num_features, index_column=None, label_column=None, rows_no_missing=None,
                 rows_one_missing=None, rows_two_missing=None, rows_three_more_missing=None, features_list=None,
                 sample_list=None):
        self.name = name
        self.num_records = num_records
        self.num_features = num_features
        self.index_column = index_column
        self.label_column = label_column
        self.rows_no_missing = rows_no_missing
        self.rows_one_missing = rows_one_missing
        self.rows_two_missing = rows_two_missing
        self.rows_three_more_missing = rows_three_more_missing
        self.features_list = features_list
        self.sample_list = sample_list

