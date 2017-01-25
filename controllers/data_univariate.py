import os

import jsonpickle
import seaborn as sns

from configuration import const_types
from configuration import paths
from controllers.data_driver import DataDriver
from model.feature import Feature
from model.features import Features


class DataUnivariate(DataDriver):
    def __init__(self, selected_dataset):
        DataDriver.__init__(self, selected_dataset)

    def generate_features_json(self):
        load_success = True

        # Check if the data file exists, and if so, load the data as needed
        if self.data is None and os.path.isfile(self.filepath):
            load_success = self.load_data()

        if load_success:

            features_collection = []
            feature_index = 0

            # Univariate analysis for each feature
            for var_name in self.data.columns.values:

                # Common for all field types
                var_datatype = self.get_data_type(var_name)
                var_vartype = self.get_variable_type(var_name)
                var_count = int(self.data[var_name].count())
                missing_count = int(self.data[var_name].isnull().sum())
                missing_percent = 100 * missing_count / float(self.data.shape[0])
                var_missing = str("%s (%.3f%%)" % (missing_count, missing_percent))
                var_unique = int(len(self.data[var_name].unique()))

                # Denote label and index, if applicable
                if self.id_column is not None and var_name == self.id_column:
                    var_vartype += " (ID)"
                elif self.label_column is not None and var_name == self.label_column:
                    var_vartype += " (Label)"

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
                var_iqr = None
                var_skew = None
                var_kurtosis = None

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
                    var_iqr = str("%.3f" % (self.data[var_name].dropna().quantile(q=0.75) -
                                            self.data[var_name].dropna().quantile(q=0.25)))
                    var_skew = str("%.3f" % self.data[var_name].skew())
                    var_kurtosis = str("%.3f" % self.data[var_name].kurt())

                    var_mode = self.get_mode(var_name)

                # Compute non-numeric stats
                else:
                    var_mostcommon = str("%s (%d)" %
                                         (self.data[var_name].value_counts().idxmax(),
                                          self.data[var_name].value_counts().max()))
                    var_leastcommon = str("%s (%d)" %
                                          (self.data[var_name].value_counts().idxmin(),
                                           self.data[var_name].value_counts().min()))

                # Histogram (numeric)
                if self.get_data_type(var_name) in [const_types.DATATYPE_FLOAT, const_types.DATATYPE_INTEGER]:
                    hist_plot = sns.distplot(self.data[var_name].dropna(), bins=None, hist=True, kde=False, rug=False)
                    graph_histogram = self.save_graph(hist_plot, var_name + paths.FILE_HISTOGRAM)

                # Countplot (non-numeric)
                elif self.check_uniques_for_graphing(var_name):
                    countplot = sns.countplot(y=self.data[var_name].dropna())
                    graph_countplot = self.save_graph(countplot, filename=var_name + paths.FILE_COUNTPLOT)

                # Save the feature stats
                feature = Feature(feat_name=var_name,
                                  feat_index=feature_index,
                                  feat_datatype=var_datatype,
                                  feat_vartype=var_vartype,
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
                                  feat_iqr=var_iqr,
                                  feat_skew=var_skew,
                                  feat_kurtosis=var_kurtosis,
                                  feat_mostcommon=var_mostcommon,
                                  feat_leastcommon=var_leastcommon,
                                  graph_histogram=graph_histogram,
                                  graph_countplot=graph_countplot)
                features_collection.append(feature)
                feature_index += 1

            # Create object holding features collection and save as JSON
            features = Features(self.title, features_collection)
            features_json = jsonpickle.encode(features)

            # Save the serialized JSON to a file
            self.save_json(json_to_write=features_json, suffix=paths.FEATURES_SUFFIX)

    def get_mode(self, feat_name):
        var_mode = ""
        mode = self.data[feat_name].mode()

        if mode is not None:
            for m in mode:
                var_mode = var_mode + str(m) + " "

        # If no mode is found, return None instead of empty string
        if var_mode == "":
            var_mode = None

        return var_mode

    def load_features_json(self):
        features_json = self.load_json(paths.FEATURES_SUFFIX)

        # If the file doesn't exist, generate it
        if features_json is None:
            self.generate_features_json()
            features_json = self.load_json(paths.FEATURES_SUFFIX)

        # Return the JSON
        return features_json
