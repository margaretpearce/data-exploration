import os
import re
import jsonpickle
import pandas as pd
import seaborn as sns

from configuration import paths
from configuration import const_types


class DataDriver:
    def __init__(self, selected_dataset):
        self.file = selected_dataset[0]
        self.title = selected_dataset[1]
        self.id_column = selected_dataset[2]
        self.label_column = selected_dataset[3]
        self.file_uploaded = selected_dataset[4]

        if self.file_uploaded:
            self.filepath = os.path.join(paths.UPLOAD_FOLDER, self.title, str(self.file))
        else:
            self.filepath = os.path.join(paths.EXAMPLES_FOLDER, self.title, str(self.file))

        self.data = None
        self.error_code = None

    def load_data(self):
        try:
            if str(self.file).endswith("csv"):
                self.data = pd.read_csv(self.filepath)
            elif str(self.file).endswith("tsv"):
                self.data = pd.read_csv(self.filepath, sep='\t')
            elif str(self.file).endswith("xls") or str(self.file).endswith("xlsx"):
                self.data = pd.read_excel(self.filepath)
            return True

        except ValueError as err:
            self.error_code = str("{0}".format(err))
            return False

    def save_graph(self, plot, filename):
        folder_path = paths.EXAMPLES_FOLDER
        relative_path = paths.EXAMPLES_RELATIVE

        if self.file_uploaded:
            folder_path = paths.UPLOAD_FOLDER
            relative_path = paths.UPLOAD_RELATIVE

        # Replace any special characters in the filename
        filename = re.sub(r'[^.a-zA-Z0-9_-]', '', filename)

        full_url = os.path.join(folder_path, self.title, str("graphs/" + filename))
        fig = plot.get_figure()
        fig.savefig(full_url)

        # Clear the figure to prepare for the next plot
        sns.plt.clf()

        # Return the relative URL to the histogram
        graph_url = relative_path + self.title + str(paths.GRAPHS_SUBFOLDER + filename)
        return graph_url

    def format_rounded_string(self, value):
        if "." in str(value):
            return str("%.3f" % float(value))
        else:
            return str(value)

    def get_data_type(self, feat_name):
        raw_type = str(self.data[feat_name].dtype)
        var_datatype = None

        # Get the data type based on its raw type
        if raw_type == "int64" or raw_type == "int8":
            # Check if it's really a boolean
            unique_vals = self.data[feat_name].unique()
            for val in unique_vals:
                if not (int(val) == 0 or int(val) == 1):
                    var_datatype = const_types.DATATYPE_INTEGER
            if not var_datatype == const_types.DATATYPE_INTEGER:
                var_datatype = const_types.DATATYPE_BOOLEAN
        elif raw_type == "bool":
            var_datatype = const_types.DATATYPE_BOOLEAN
        elif raw_type == "float64" or raw_type == "float32":
            var_datatype = const_types.DATATYPE_FLOAT
        elif raw_type == "datetime64":
            var_datatype = const_types.DATATYPE_DATE
        elif raw_type == "object":
            var_datatype = const_types.DATATYPE_STRING

        return var_datatype

    def get_variable_type(self, feat_name):
        # Get the variable type based on data type and heuristics
        var_datatype = self.get_data_type(feat_name)
        var_vartype = const_types.VARTYPE_UNKNOWN

        # Variable type: categorical, continuous, binary
        if var_datatype == const_types.DATATYPE_BOOLEAN:
            var_vartype = const_types.VARTYPE_BINARY
        elif var_datatype == const_types.DATATYPE_STRING or var_datatype == const_types.DATATYPE_DATE:
            var_vartype = const_types.VARTYPE_CATEGORICAL
        elif var_datatype == const_types.DATATYPE_INTEGER or var_datatype == const_types.DATATYPE_FLOAT:
            # Distinguish int categorical variables (e.g. 1, 2, 3) from int continuous (e.g. 1, 2, ..., 1000)
            # Assuming categorical features have 10% or fewer unique values over the data set
            if self.get_percent_unique(feat_name) < 0.10:
                var_vartype = const_types.VARTYPE_CATEGORICAL
            else:
                var_vartype = const_types.VARTYPE_CONTINUOUS

        return var_vartype

    def get_percent_unique(self, feat_name):
        return float(len(self.data[feat_name].unique())) / self.data[feat_name].count()

    def get_count_unique(self, feat_name):
        return len(self.data[feat_name].unique())

    def check_uniques_for_graphing(self, feat_name):
        return self.get_percent_unique(feat_name) < 0.2 or self.get_count_unique(feat_name) < 12

    def get_error_msg(self):
        return self.error_code

    def save_json(self, json_to_write, suffix):
        folder_path = paths.EXAMPLES_FOLDER

        if self.file_uploaded:
            folder_path = paths.UPLOAD_FOLDER

        file = open(os.path.join(folder_path, self.title, suffix), 'w')
        file.write(json_to_write)
        file.close()

    def load_json(self, json_suffix):
        folder_path = paths.EXAMPLES_FOLDER

        if self.file_uploaded:
            folder_path = paths.UPLOAD_FOLDER

        absolute_filename = os.path.join(folder_path, self.title, json_suffix)

        # Check if the JSON file exists and if not, generate it
        if not os.path.isfile(absolute_filename):
            return None

        # Read serialized JSON file
        if os.path.isfile(absolute_filename):
            with open(absolute_filename, 'r') as serialized_file:
                json_str = serialized_file.read()
                deserialized_json = jsonpickle.decode(json_str)
            return deserialized_json
