import pandas as pd
import os
import json
import jsonpickle
import paths
from DataSummary import DataSummary


class DataDriver:
    def __init__(self, datafile, title, idcolumn=None, labelcolumn=None):
        self.file = datafile
        self.filepath = os.path.join(paths.EXAMPLES_FOLDER, self.file)
        self.title = title
        self.id_column = idcolumn
        self.label_column = labelcolumn

        # Other class variables
        self.data = None

        # Check if the JSON files exist, and if not, load the data and generate them
        if os.path.isfile(self.filepath):
            self.load_file()
            self.generate_summary_json()
            self.generate_features_json()
            self.generate_interactions_json()

    def load_file(self):
        # Load the data into a Pandas dataframe
        if str(self.file).endswith("csv"):
            self.data = pd.read_csv(self.filepath)
        elif str(self.file).endswith("xls") or str(self.file).endswith("xlsx"):
            self.data = pd.read_excel(self.filepath)
        elif str(self.file).endswith("json"):
            self.data = pd.read_json(self.filepath)

    def generate_summary_json(self):
        # Get summary stats about the data and serialize it as JSON
        summary = DataSummary(self.title, self.data.shape[0])
        summary_json = jsonpickle.encode(summary)

        # Save the serialized JSON to a file
        file = open(os.path.join(paths.EXAMPLES_FOLDER, str(self.title + "_summary.json")), 'w')
        file.write(summary_json)
        file.close()

    def generate_features_json(self):
        return None

    def generate_interactions_json(self):
        return None

    def load_summary_json(self):
        # Read serialized JSON file
        with open(os.path.join(paths.EXAMPLES_FOLDER, str(self.title + "_summary.json")), 'r') as summary_file:
            summary_str = summary_file.read()
            json_file = jsonpickle.decode(summary_str)
        return json_file

    def load_features_json(self):
        with open(os.path.join(paths.EXAMPLES_FOLDER, str(self.title + "_features.json")), 'r') as features_file:
            data = json.load(features_file)
        return data

    def load_interactions_json(self):
        with open(os.path.join(paths.EXAMPLES_FOLDER, str(self.title + "_interactions.json")), 'r') as interaction_file:
            data = json.load(interaction_file)
        return data


