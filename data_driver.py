import pandas as pd
import os
import jsonpickle
import paths
import seaborn as sns
from Summary import DataSummary
from Interactions import Interactions
from Feature import Feature
from Features import Features

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

        self.numeric_data = self.data.select_dtypes(include=['int64', 'float64'])
        self.numeric_fieldnames = list(self.numeric_data.columns.values)
        if self.id_column in self.numeric_fieldnames:
            self.numeric_fieldnames.remove(self.id_column)

    def load_data(self):
        # Load the data into a Pandas DataFrame
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
        index_column = self.id_column
        label_column = self.label_column

        # Count the number of columns missing for each row
        count_missing = self.data.apply(lambda x: sum(x.isnull().values), axis=1)
        self.data["num_missing"] = pd.Series(count_missing)

        num_rows_no_missing = int(sum(self.data["num_missing"] == 0))
        num_rows_one_missing = int(sum(self.data["num_missing"] == 1))
        num_rows_two_missing = int(sum(self.data["num_missing"] == 2))
        num_rows_three_more_missing = int(sum(self.data["num_missing"] >= 3))

        # Sample data (five rows)
        features_list = list(self.data.columns.values)
        sample_list = self.data.head()[features_list].values.tolist()

        summary = DataSummary(name=self.title,
                              num_records=num_records,
                              num_features=num_features,
                              index_column=index_column,
                              label_column=label_column,
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
        features_collection = []
        feature_index = 0

        # For each feature, get as much relevant info as possible
        for var_name in self.data.columns.values:
            # Common for all field types
            var_type = None
            raw_type = str(self.data[var_name].dtype)

            if raw_type == "int64":
                # Check if it's really a boolean
                unique_vals = self.data[var_name].unique()
                for val in unique_vals:
                    if not (int(val)==0 or int(val)==1):
                        var_type = "Integer"
                if not var_type == "Integer":
                    var_type = "Boolean"
            elif raw_type == "float64":
                var_type = "Float"
            elif raw_type == "datetime64":
                var_type = "Date"
            elif raw_type == "object":
                var_type = "String"

            var_count = int(self.data[var_name].count())

            missing_count = int(self.data[var_name].isnull().sum())
            missing_percent = missing_count / float(var_count)
            var_missing = str("%s (%.3f%%)" % (missing_count, missing_percent))

            var_unique = int(len(self.data[var_name].unique()))

            # Numeric only
            var_avg = None
            var_median = None
            var_mode = None
            var_max = None
            var_min = None
            var_stddev = None
            var_variance = None
            var_quantile25 = None
            var_quantile75 = None

            # Non-numeric only
            var_mostcommon = None
            var_leastcommon = None

            # Graphs
            graph_histogram = None
            graph_countplot = None

            # Compute numeric statistics
            if self.data[var_name].dtype in ['int64', 'float64']:
                var_avg = str("%.3f" % float(self.data[var_name].mean()))
                var_median = float(self.data[var_name].median())
                var_max = float(self.data[var_name].max())
                var_min = float(self.data[var_name].min())
                var_stddev = str("%.3f" % self.data[var_name].std())
                var_variance = str("%.3f" % self.data[var_name].var())
                var_quantile25 = str("%.3f" % self.data[var_name].dropna().quantile(q=0.25))
                var_quantile75 = str("%.3f" % self.data[var_name].dropna().quantile(q=0.75))

                mode = self.data[var_name].mode()
                if not mode is None:
                    var_mode = ""
                    for m in mode:
                        var_mode = var_mode + str(m)

            # Compute non-numeric stats
            else:
                var_mostcommon = str("%s (%d)" %
                                     (self.data[var_name].value_counts().idxmax(),
                                     self.data[var_name].value_counts().max()))
                var_leastcommon = str("%s (%d)" %
                                     (self.data[var_name].value_counts().idxmin(),
                                     self.data[var_name].value_counts().min()))

            # Histogram/ countplot
            if self.data[var_name].dtype in ['int64', 'float64']:
                hist_plot = sns.distplot(self.data[var_name].dropna(), bins=None, hist=True, kde=False, rug=False)
                full_url = os.path.join(paths.EXAMPLES_FOLDER, str(var_name + "_hist.png"))
                fig = hist_plot.get_figure()
                fig.savefig(full_url)   # Save the histogram
                sns.plt.clf()   # Clear the figure to prepare for the next plot
                graph_histogram = paths.EXAMPLES_RELATIVE + str(var_name + "_hist.png")    # Relative histogram URL
            else:
                countplot = sns.countplot(y=self.data[var_name].dropna())

                if (var_unique / float(var_count)) > 0.5:
                    countplot.set(ylabel='')
                    countplot.set(yticklabels=[])
                    countplot.yaxis.set_visible(False)

                full_url = os.path.join(paths.EXAMPLES_FOLDER, str(var_name + "_countplot.png"))
                fig = countplot.get_figure()
                fig.savefig(full_url)
                sns.plt.clf()   # Clear the figure to prepare for the next plot

                # Return the relative URL to the histogram
                graph_countplot = paths.EXAMPLES_RELATIVE + str(var_name + "_countplot.png")

            feature = Feature(feat_name=var_name,
                              feat_index=feature_index,
                              feat_type=var_type,
                              feat_count=var_count,
                              feat_missing=var_missing,
                              feat_unique=var_unique,
                              feat_average=var_avg,
                              feat_median=var_median,
                              feat_mode=var_mode,
                              feat_max=var_max,
                              feat_min=var_min,
                              feat_stddev=var_stddev,
                              feat_variance=var_variance,
                              feat_quantile25=var_quantile25,
                              feat_quantile75=var_quantile75,
                              feat_mostcommon=var_mostcommon,
                              feat_leastcommon=var_leastcommon,
                              graph_histogram=graph_histogram,
                              graph_countplot=graph_countplot)
            features_collection.append(feature)
            feature_index = feature_index + 1

        # Create object holding features collection and save as JSON
        features = Features(self.title, features_collection)
        features_json = jsonpickle.encode(features)

        # Save the serialized JSON to a file
        file = open(os.path.join(paths.EXAMPLES_FOLDER, str(self.title + FEATURES_SUFFIX)), 'w')
        file.write(features_json)
        file.close()

    def generate_interactions_json(self):
        # Get correlations between features
        correlations = self.numeric_data.corr()
        correlation_heatmap = sns.heatmap(correlations, vmax=1, square=True)

        # Save the heatmap
        correlation_url = os.path.join(paths.EXAMPLES_FOLDER, str(self.title + "_corr.png"))
        correlation_url_relative = paths.EXAMPLES_RELATIVE + str(self.title + "_corr.png")
        fig = correlation_heatmap.get_figure()
        fig.savefig(correlation_url)

        # Clear the figure to prepare for the next plot
        sns.plt.clf()

        # Get covariance between features
        covariance = self.data.cov()
        covariance_heatmap = sns.heatmap(covariance, vmax=1, square=True)
        covariance_url = os.path.join(paths.EXAMPLES_FOLDER, str(self.title + "_cov.png"))
        covariance_url_relative = paths.EXAMPLES_RELATIVE + str(self.title + "_cov.png")
        fig = covariance_heatmap.get_figure()
        fig.savefig(covariance_url)
        sns.plt.clf()

        # Save data as JSON
        interactions = Interactions(name=self.title,
                                    correlations_url=correlation_url_relative,
                                    covariance_url=covariance_url_relative)
        interactions_json = jsonpickle.encode(interactions)

        # Save the serialized JSON to a file
        file = open(os.path.join(paths.EXAMPLES_FOLDER, str(self.title + INTERACTIONS_SUFFIX)), 'w')
        file.write(interactions_json)
        file.close()

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



