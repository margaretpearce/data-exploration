import os
import jsonpickle
import pandas as pd
from collections import OrderedDict

from data_driver import DataDriver
from model.summary import Summary
from configuration import paths


class DataSummary(DataDriver):
    def __init__(self, selected_dataset):
        DataDriver.__init__(self, selected_dataset)

    def generate_summary_json(self):
        load_success = True

        # Check if the data file exists, and if so, load the data as needed
        if self.data is None and os.path.isfile(self.filepath):
            load_success = self.load_data()

        if load_success:
            # Get summary stats about the data and serialize it as JSON
            num_records = self.data.shape[0]
            num_features = self.data.shape[1]
            index_column = self.id_column
            label_column = self.label_column

            # Count the number of columns missing for each row
            num_rows_missing = self.count_missing(num_records)

            # List of features
            features_list = list(self.data.columns.values)

            # Sample data (five rows or less)
            sample_list = self.get_sample(num_records, features_list)

            summary = Summary(name=self.title,
                              num_records=num_records,
                              num_features=num_features,
                              index_column=index_column,
                              label_column=label_column,
                              rows_missing=num_rows_missing,
                              features_list=features_list,
                              sample_list=sample_list
                              )
            summary_json = jsonpickle.encode(summary)

            # Save the serialized JSON to a file
            self.save_json(json_to_write=summary_json, suffix=paths.SUMMARY_SUFFIX)

    def count_missing(self, num_records):
        # Count the number of columns missing for each row
        count_missing = self.data.apply(lambda x: sum(x.isnull().values), axis=1)
        self.data["num_missing"] = pd.Series(count_missing)

        num_rows_missing = {}
        cumulative_row_sum = 0

        for i in range(0, num_records):
            # Count the number of missing rows and add it to the list
            num_rows_i_missing = int(sum(self.data["num_missing"] == i))
            num_rows_missing[i] = num_rows_i_missing
            cumulative_row_sum += num_rows_i_missing

            # If we reached the point where we have accounted for all rows, stop looking for rows
            if cumulative_row_sum == num_records:
                break

        # Sort the dictionary
        num_rows_missing = OrderedDict(sorted(num_rows_missing.items()))

        # Remove the added column
        self.data.drop("num_missing", axis=1, inplace=True)

        return num_rows_missing

    def get_sample(self, num_records, features_list):
        num_samples = 5
        if num_records < num_samples:
            num_samples = num_records

        sample_list = self.data.sample(num_samples)[features_list].values.tolist()
        return sample_list

    def load_summary_json(self):
        # Try to load an existing JSON file
        summary_json = self.load_json(paths.SUMMARY_SUFFIX)

        # If the file doesn't exist, generate it
        if summary_json is None:
            self.generate_summary_json()
            summary_json = self.load_json(paths.SUMMARY_SUFFIX)

        # Return the JSON
        return summary_json
