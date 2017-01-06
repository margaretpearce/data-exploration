class Summary(object):
    def __init__(self, name, num_records, num_features, index_column=None, label_column=None, rows_missing = None,
                 features_list=None, sample_list=None):
        self.name = name
        self.num_records = num_records
        self.num_features = num_features
        self.index_column = index_column
        self.label_column = label_column
        self.rows_missing = rows_missing
        self.features_list = features_list
        self.sample_list = sample_list

