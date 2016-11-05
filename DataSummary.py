class DataSummary(object):
    def __init__(self, name, numrecords, numfeatures, rows_no_missing):
        self.name = name
        self.numrecords = numrecords
        self.numfeatures = numfeatures
        self.rows_no_missing = rows_no_missing
