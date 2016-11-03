import pandas as pd
import os, json
import paths

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
        if not os.path.isfile(self.filepath):
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
        # Create empty dictionary (JSON base)
        data_summary = {}

        # Save a single data point
        data_summary["numrecords"] = 800

        # Get lists of other data points to be displayed in sequence (e.g. in a table)
        headers =  ["Total # records", "Records with no missing data", "Records with 1 column missing data"]
        values = [self.data.shape[0], 500, 132]

        # Save lists as nested JSON
        data_summary["summary"] = []
        for i in range(len(headers)):
            row = {}
            row["name"] = headers[i]
            row["value"] = values[i]
            data_summary["summary"].append(row)
        json_summary = json.dumps(data_summary)

        # Save the JSON file
        with open (os.path.join(paths.EXAMPLES_FOLDER, str(self.title + "_summary.json")), 'w') as outfile:
            json.dump(json_summary, outfile)

    def generate_features_json(self):
        return None

    def generate_interactions_json(self):
        return None

    def load_summary_json(self):
        with open(os.path.join(paths.EXAMPLES_FOLDER, str(self.title + "_summary.json")), 'r') as summary_file:
            data = json.load(summary_file)
        return data

    def load_features_json(self):
        with open(os.path.join(paths.EXAMPLES_FOLDER, str(self.title + "_features.json")), 'r') as features_file:
            data = json.load(features_file)
        return data

    def load_interactions_json(self):
        with open(os.path.join(paths.EXAMPLES_FOLDER, str(self.title + "_interactions.json")), 'r') as interactions_file:
            data = json.load(interactions_file)
        return data

