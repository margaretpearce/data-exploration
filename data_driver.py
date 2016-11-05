import pandas as pd
import os
import json
import jsonpickle
import paths
from DataSummary import DataSummary

SUMMARY_SUFFIX = "_summary.json"
FEATURES_SUFFIX = "_features.json"
INTERACTIONS_SUFFIX = "_interactions.json"

class DataDriver:
    def __init__(self, datafile, title, idcolumn=None, labelcolumn=None):
        self.file = datafile
        self.filepath = os.path.join(paths.EXAMPLES_FOLDER, self.file)
        self.title = title
        self.id_column = idcolumn
        self.label_column = labelcolumn

        # Other class variables
        self.data = None

        # Check if the data file exists, and if so, load the data
        if os.path.isfile(self.filepath):
            self.load_data()

    def load_data(self):
        # Load the data into a Pandas dataframe
        if str(self.file).endswith("csv"):
            self.data = pd.read_csv(self.filepath)
        elif str(self.file).endswith("xls") or str(self.file).endswith("xlsx"):
            self.data = pd.read_excel(self.filepath)
        elif str(self.file).endswith("json"):
            self.data = pd.read_json(self.filepath)

    def generate_summary_json(self):
        # Get summary stats about the data and serialize it as JSON
        num_records = self.data.shape[0]
        num_features = self.data.shape[1]

        # Count the number of columns missing for each row
        count_missing = self.data.apply(lambda x: sum(x.isnull().values), axis = 1)
        self.data["num_missing"] = pd.Series(count_missing)

        num_rows_no_missing = int(sum(self.data["num_missing"] == 0))
        num_rows_one_missing = int(sum(self.data["num_missing"] == 1))
        num_rows_two_missing = int(sum(self.data["num_missing"] == 2))
        num_rows_three_more_missing = int(sum(self.data["num_missing"] >= 3))

        # Sample data (five rows)
        features_list = list(self.data.columns.values)
        sample_list = self.data.head()[features_list].values.tolist()
        # sample_list = None

        summary = DataSummary(self.title,
                              num_records=num_records,
                              num_features=num_features,
                              rows_no_missing=num_rows_no_missing,
                              rows_one_missing=num_rows_one_missing,
                              rows_two_missing=num_rows_two_missing,
                              rows_three_more_missing=num_rows_three_more_missing,
                              features_list=features_list,
                              sample_list=sample_list
                              )
        summary_json = jsonpickle.encode(summary)

        # Save the serialized JSON to a file
        file = open(os.path.join(paths.EXAMPLES_FOLDER, str(self.title + SUMMARY_SUFFIX)), 'w')
        file.write(summary_json)
        file.close()

    def generate_features_json(self):
        return None

    def generate_interactions_json(self):
        return None

    def load_summary_json(self):
        return self.load_json(SUMMARY_SUFFIX)

    def load_features_json(self):
        return self.load_json(FEATURES_SUFFIX)

    def load_interactions_json(self):
        return self.load_json(INTERACTIONS_SUFFIX)

    def load_json(self, json_suffix):
        absolute_filename = os.path.join(paths.EXAMPLES_FOLDER, str(self.title + json_suffix))

        # Check if the JSON file exists and if not, generate it
        if not os.path.isfile(absolute_filename):
            if json_suffix == SUMMARY_SUFFIX:
                self.generate_summary_json()
            elif json_suffix == FEATURES_SUFFIX:
                self.generate_features_json()
            elif json_suffix == INTERACTIONS_SUFFIX:
                self.generate_interactions_json()

        # Read serialized JSON file
        with open(absolute_filename, 'r') as serialized_file:
            json_str = serialized_file.read()
            deserialized_json = jsonpickle.decode(json_str)
        return deserialized_json


